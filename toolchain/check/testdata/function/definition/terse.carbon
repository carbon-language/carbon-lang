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

fn GenericNonDependent[T: type](unused x: T) => 42 as i32;

fn TestGenericNonDependent() {
  let unused a: i32 = GenericNonDependent(true);
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

// --- main_run_i32.carbon
library "[[@TEST_NAME]]";

fn Run() => 0 as i32;

// --- main_run_unit.carbon
library "[[@TEST_NAME]]";

fn Run() => ();

// --- fail_main_run_bool.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_main_run_bool.carbon:[[@LINE+4]]:1: error: invalid return type for `Main.Run` function; expected `fn (...)` or `fn (...) -> i32` [InvalidMainRunReturnType]
// CHECK:STDERR: fn Run() => true;
// CHECK:STDERR: ^~~~~~~~~~~
// CHECK:STDERR:
fn Run() => true;

// --- fail_redecl.carbon
library "[[@TEST_NAME]]";

fn DeclaredI32() -> i32;
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `auto` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn DeclaredI32() => 0 as i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared with return type `i32` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR: fn DeclaredI32() -> i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn DeclaredI32() => 0 as i32;

fn DeclaredUnit();
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `auto` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn DeclaredUnit() => ();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
// CHECK:STDERR: fn DeclaredUnit();
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn DeclaredUnit() => ();

fn TerseFirst() => 0 as i32;
// CHECK:STDERR: fail_redecl.carbon:[[@LINE+7]]:1: error: function redeclaration differs because return type is `auto` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fn TerseFirst() => 1 as i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_redecl.carbon:[[@LINE-4]]:1: note: previously declared with return type `i32` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR: fn TerseFirst() => 0 as i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn TerseFirst() => 1 as i32;

// --- fail_incomplete_return.carbon
library "[[@TEST_NAME]]";

class Incomplete;

// CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE+7]]:39: error: function returns incomplete type `Incomplete` [IncompleteTypeInFunctionReturnType]
// CHECK:STDERR: fn DerefIncomplete(p: Incomplete*) => *p;
// CHECK:STDERR:                                       ^~
// CHECK:STDERR: fail_incomplete_return.carbon:[[@LINE-5]]:1: note: class was forward declared here [ClassForwardDeclaredHere]
// CHECK:STDERR: class Incomplete;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn DerefIncomplete(p: Incomplete*) => *p;

// --- fail_unused.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_unused.carbon:[[@LINE+11]]:15: warning: binding `x` unused [UnusedBinding]
// CHECK:STDERR: fn UnusedTest(x: i32, unused y: i32) => y;
// CHECK:STDERR:               ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_unused.carbon:[[@LINE+7]]:30: error: variable `y` marked `unused` but used [UnusedButUsed]
// CHECK:STDERR: fn UnusedTest(x: i32, unused y: i32) => y;
// CHECK:STDERR:                              ^~~~~~
// CHECK:STDERR: fail_unused.carbon:[[@LINE+4]]:41: note: usage here [UnusedButUsedHere]
// CHECK:STDERR: fn UnusedTest(x: i32, unused y: i32) => y;
// CHECK:STDERR:                                         ^
// CHECK:STDERR:
fn UnusedTest(x: i32, unused y: i32) => y;

// --- fail_todo_dependent_return.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_dependent_return.carbon:[[@LINE+4]]:1: error: semantics TODO: `deduced return type that depends on a generic parameter` [SemanticsTodo]
// CHECK:STDERR: fn GenericId[T: Core.Copy](x: T) => x;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn GenericId[T: Core.Copy](x: T) => x;

// --- fail_todo_call_before_deduced.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_call_before_deduced.carbon:[[@LINE+4]]:23: error: semantics TODO: `call to function before return type is deduced` [SemanticsTodo]
// CHECK:STDERR: fn Recurse(x: i32) => Recurse(x);
// CHECK:STDERR:                       ^~~~~~~~~~
// CHECK:STDERR:
fn Recurse(x: i32) => Recurse(x);

class EarlyCall {
  fn M() => 1 as i32;
  // CHECK:STDERR: fail_todo_call_before_deduced.carbon:[[@LINE+4]]:16: error: semantics TODO: `call to function before return type is deduced` [SemanticsTodo]
  // CHECK:STDERR:   var x: i32 = M();
  // CHECK:STDERR:                ^~~
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
    // CHECK:STDERR: fail_todo_thunk_before_deduced.carbon:[[@LINE+4]]:5: error: semantics TODO: `thunk for function before return type is deduced` [SemanticsTodo]
    // CHECK:STDERR:     fn F() => 0 as i32;
    // CHECK:STDERR:     ^~~~~~~~~
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
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
// CHECK:STDOUT:   %Id.type: type = fn_type @Id [concrete]
// CHECK:STDOUT:   %Id: %Id.type = struct_value () [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Unit.type: type = fn_type @Unit [concrete]
// CHECK:STDOUT:   %Unit: %Unit.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %y.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.6b6 = wrapper_binding_pattern y, %y.param_patt [concrete]
// CHECK:STDOUT:   %Pair.type: type = fn_type @Pair [concrete]
// CHECK:STDOUT:   %Pair: %Pair.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [concrete]
// CHECK:STDOUT:   %.7fe: Core.Form = init_form %struct_type.a.b [concrete]
// CHECK:STDOUT:   %pattern_type.20e3: type = pattern_type %struct_type.a.b [concrete]
// CHECK:STDOUT:   %return.param_patt.3a9: %pattern_type.20e3 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.1de: %pattern_type.20e3 = return_slot_pattern %return.param_patt.3a9, %struct_type.a.b [concrete]
// CHECK:STDOUT:   %TestBasic.type: type = fn_type @TestBasic [concrete]
// CHECK:STDOUT:   %TestBasic: %TestBasic.type = struct_value () [concrete]
// CHECK:STDOUT:   %a.patt.bb2: %pattern_type.6b6 = value_binding_pattern a [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.48f: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet.48f) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet.48f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf3: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.098: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.ac4: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %p.patt: %pattern_type.20e3 = value_binding_pattern p [concrete]
// CHECK:STDOUT:   %int_10.64f: Core.IntLiteral = int_value 10 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcb: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.1b6a: <bound method> = bound_method %int_10.64f, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_10.6bf: %i32 = int_value 10 [concrete]
// CHECK:STDOUT:   %b.patt.e7d61a.1: %pattern_type.6b6 = value_binding_pattern b [concrete]
// CHECK:STDOUT:   %self.param_patt.705: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.70d: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt.705 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc13_41.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc13_41.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.1: type = fn_type @Destroy.WithSelf.SelfDestruct.loc13_41.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.1: %Destroy.WithSelf.SelfDestruct.type.fbceb5.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.3bf: %pattern_type.20e3 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d0e: %pattern_type.20e3 = wrapper_binding_pattern self, %self.param_patt.3bf [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc13_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc13_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc13_41.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc13_41.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc13_41.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc13_41.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.20e3 = ref_param_pattern [concrete = constants.%self.param_patt.3bf]
// CHECK:STDOUT:     %self.patt: %pattern_type.20e3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d0e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc13_41.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.20e3 = ref_param_pattern [concrete = constants.%self.param_patt.3bf]
// CHECK:STDOUT:     %self.patt: %pattern_type.20e3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d0e]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Id.decl: %Id.type = fn_decl @Id [concrete = constants.%Id] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt.cee]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.e3b]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, constants.%i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc4: Core.Form = init_form constants.%i32 [concrete = constants.%.795f]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Unit.decl: %Unit.type = fn_decl @Unit [concrete = constants.%Unit] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, constants.%empty_tuple.type [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_15.4: Core.Form = init_form constants.%empty_tuple.type [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Pair.decl: %Pair.type = fn_decl @Pair [concrete = constants.%Pair] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt.cee]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.e3b]
// CHECK:STDOUT:     %y.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%y.param_patt]
// CHECK:STDOUT:     %y.patt: %pattern_type.6b6 = wrapper_binding_pattern y, %y.param_patt [concrete = constants.%y.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.20e3 = out_param_pattern [concrete = constants.%return.param_patt.3a9]
// CHECK:STDOUT:     %return.patt: %pattern_type.20e3 = return_slot_pattern %return.param_patt, constants.%struct_type.a.b [concrete = constants.%return.patt.1de]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc8_12: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:     %y.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32.loc8_20: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %y: %i32 = wrapper_binding y, %y.param
// CHECK:STDOUT:     %.loc8_43.8: Core.Form = init_form constants.%struct_type.a.b [concrete = constants.%.7fe]
// CHECK:STDOUT:     %return.param: ref %struct_type.a.b = out_param call_param2
// CHECK:STDOUT:     %return: ref %struct_type.a.b = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TestBasic.decl: %TestBasic.type = fn_decl @TestBasic [concrete = constants.%TestBasic] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Id(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc4_18.1: <bound method> = bound_method %x.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_18.2: <bound method> = bound_method %x.ref, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc4_18.2(%x.ref)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Unit() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_15.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_15.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc6_15.3: init %empty_tuple.type = converted %.loc6_15.1, %.loc6_15.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc6_15.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Pair(%x.param: %i32, %y.param: %i32) -> out %return.param: %struct_type.a.b {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %y.ref: %i32 = name_ref y, %y
// CHECK:STDOUT:   %.loc8_43.1: %struct_type.a.b = struct_literal (%x.ref, %y.ref)
// CHECK:STDOUT:   %impl.elem0.loc8_34: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %x.ref, %impl.elem0.loc8_34
// CHECK:STDOUT:   %specific_fn.loc8_34: <specific function> = specific_function %impl.elem0.loc8_34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %x.ref, %specific_fn.loc8_34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc8_34: init %i32 = call %bound_method.loc8_34.2(%x.ref)
// CHECK:STDOUT:   %.loc8_43.2: ref %i32 = struct_access %return.param, element0
// CHECK:STDOUT:   %.loc8_43.3: init %i32 to %.loc8_43.2 = in_place_init %Int.as.Copy.impl.Op.call.loc8_34
// CHECK:STDOUT:   %impl.elem0.loc8_42: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc8_42.1: <bound method> = bound_method %y.ref, %impl.elem0.loc8_42
// CHECK:STDOUT:   %specific_fn.loc8_42: <specific function> = specific_function %impl.elem0.loc8_42, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_42.2: <bound method> = bound_method %y.ref, %specific_fn.loc8_42
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc8_42: init %i32 = call %bound_method.loc8_42.2(%y.ref)
// CHECK:STDOUT:   %.loc8_43.4: ref %i32 = struct_access %return.param, element1
// CHECK:STDOUT:   %.loc8_43.5: init %i32 to %.loc8_43.4 = in_place_init %Int.as.Copy.impl.Op.call.loc8_42
// CHECK:STDOUT:   %.loc8_43.6: init %struct_type.a.b to %return.param = struct_init (%.loc8_43.3, %.loc8_43.5)
// CHECK:STDOUT:   %.loc8_43.7: init %struct_type.a.b = converted %.loc8_43.1, %.loc8_43.6
// CHECK:STDOUT:   return %.loc8_43.7 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TestBasic() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Id.ref: %Id.type = name_ref Id, file.%Id.decl [concrete = constants.%Id]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %impl.elem0.loc11: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc11_19.1: <bound method> = bound_method %int_42, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.bf3]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_19.2: <bound method> = bound_method %int_42, %specific_fn.loc11 [concrete = constants.%bound_method.098]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11: init %i32 = call %bound_method.loc11_19.2(%int_42) [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc11_19.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc11 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc11_19.2: %i32 = converted %int_42, %.loc11_19.1 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %Id.call: init %i32 = call %Id.ref(%.loc11_19.2)
// CHECK:STDOUT:   %.loc11_21.1: %i32 = value_of_initializer %Id.call
// CHECK:STDOUT:   %.loc11_21.2: %i32 = converted %Id.call, %.loc11_21.1
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: %i32 = wrapper_binding a, %.loc11_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6b6 = value_binding_pattern a [concrete = constants.%a.patt.bb2]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Unit.ref: %Unit.type = name_ref Unit, file.%Unit.decl [concrete = constants.%Unit]
// CHECK:STDOUT:   %Unit.call: init %empty_tuple.type = call %Unit.ref()
// CHECK:STDOUT:   %Pair.ref: %Pair.type = name_ref Pair, file.%Pair.decl [concrete = constants.%Pair]
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [concrete = constants.%int_10.64f]
// CHECK:STDOUT:   %.loc13_41.1: ref %struct_type.a.b = temporary_storage
// CHECK:STDOUT:   %impl.elem0.loc13: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_39.1: <bound method> = bound_method %int_10, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.dcb]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_39.2: <bound method> = bound_method %int_10, %specific_fn.loc13 [concrete = constants.%bound_method.1b6a]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13: init %i32 = call %bound_method.loc13_39.2(%int_10) [concrete = constants.%int_10.6bf]
// CHECK:STDOUT:   %.loc13_39.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc13 [concrete = constants.%int_10.6bf]
// CHECK:STDOUT:   %.loc13_39.2: %i32 = converted %int_10, %.loc13_39.1 [concrete = constants.%int_10.6bf]
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
// CHECK:STDOUT:     %p.patt: %pattern_type.20e3 = value_binding_pattern p [concrete = constants.%p.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %p.ref: %struct_type.a.b = name_ref p, %p
// CHECK:STDOUT:   %.loc14: %i32 = struct_access %p.ref, element0
// CHECK:STDOUT:   %i32.loc14: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: %i32 = wrapper_binding b, %.loc14
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.6b6 = value_binding_pattern b [concrete = constants.%b.patt.e7d61a.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc13_41.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc13_41.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc13_41.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_41.1(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc13_41.1(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc13_41.2(%self.param: ref %struct_type.a.b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc13_41.1: ref %i32 = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc13_41.1: <bound method> = bound_method %.loc13_41.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc13_41.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc13_41.1(%.loc13_41.1)
// CHECK:STDOUT:   %.loc13_41.2: ref %i32 = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc13_41.2: <bound method> = bound_method %.loc13_41.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc13_41.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc13_41.2(%.loc13_41.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc13_41.2(%self.param: ref %struct_type.a.b) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc13_41.2(%self.param: ref %struct_type.a.b) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

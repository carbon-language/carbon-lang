// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/fn_ptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/fn_ptr.carbon

// --- fn_ptr_call.carbon
library "[[@TEST_NAME]]";

fn A(x: i32) -> i32;

fn F() {
  //@dump-sem-ir-begin
  var a_ptr: __fn_ptr(form(val i32)) ->? form(var i32) = A;
  var unused result: i32 = a_ptr(0);
  //@dump-sem-ir-end
}

// --- fail_fn_ptr_call_mismatch.carbon
library "[[@TEST_NAME]]";

fn A(x: i32) -> i32;

fn F() {
  var a_ptr: __fn_ptr(form(val i32)) ->? form(var i32) = A;
  // CHECK:STDERR: fail_fn_ptr_call_mismatch.carbon:[[@LINE+7]]:34: error: cannot implicitly convert expression of type `Core.FloatLiteral` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var unused result: i32 = a_ptr(1.0);
  // CHECK:STDERR:                                  ^~~
  // CHECK:STDERR: fail_fn_ptr_call_mismatch.carbon:[[@LINE+4]]:34: note: type `Core.FloatLiteral` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused result: i32 = a_ptr(1.0);
  // CHECK:STDERR:                                  ^~~
  // CHECK:STDERR:
  var unused result: i32 = a_ptr(1.0);
}

// --- fail_param_category_mismatch.carbon
library "[[@TEST_NAME]]";

fn A(x: i32) -> i32;

fn F() {
  // CHECK:STDERR: fail_param_category_mismatch.carbon:[[@LINE+7]]:65: error: function pointer form does not match function declaration [FunctionPtrFormMismatch]
  // CHECK:STDERR:   var unused a_ptr: __fn_ptr(form(ref i32)) ->? form(var i32) = A;
  // CHECK:STDERR:                                                                 ^
  // CHECK:STDERR: fail_param_category_mismatch.carbon:[[@LINE-6]]:6: note: this parameter/return's form is not form(ref i32) [FunctionPtrFormMismatchParamPattern]
  // CHECK:STDERR: fn A(x: i32) -> i32;
  // CHECK:STDERR:      ^~~~~~
  // CHECK:STDERR:
  var unused a_ptr: __fn_ptr(form(ref i32)) ->? form(var i32) = A;
}

// --- fail_return_category_mismatch.carbon
library "[[@TEST_NAME]]";

fn A(x: i32) -> i32;

fn F() {
  // CHECK:STDERR: fail_return_category_mismatch.carbon:[[@LINE+7]]:65: error: function pointer form does not match function declaration [FunctionPtrFormMismatch]
  // CHECK:STDERR:   var unused a_ptr: __fn_ptr(form(val i32)) ->? form(ref i32) = A;
  // CHECK:STDERR:                                                                 ^
  // CHECK:STDERR: fail_return_category_mismatch.carbon:[[@LINE-6]]:17: note: this parameter/return's form is not form(ref i32) [FunctionPtrFormMismatchParamPattern]
  // CHECK:STDERR: fn A(x: i32) -> i32;
  // CHECK:STDERR:                 ^~~
  // CHECK:STDERR:
  var unused a_ptr: __fn_ptr(form(val i32)) ->? form(ref i32) = A;
}

// --- fail_param_type_mismatch.carbon
library "[[@TEST_NAME]]";

fn A(x: i32) -> i32;

fn F() {
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE+7]]:65: error: type `i64` in function pointer type differs from type `i32` in function signature [FunctionPtrDiffersType]
  // CHECK:STDERR:   var unused a_ptr: __fn_ptr(form(val i64)) ->? form(var i32) = A;
  // CHECK:STDERR:                                                                 ^
  // CHECK:STDERR: fail_param_type_mismatch.carbon:[[@LINE-6]]:6: note: declared in function signature here [FunctionPtrDiffersPrevious]
  // CHECK:STDERR: fn A(x: i32) -> i32;
  // CHECK:STDERR:      ^~~~~~
  // CHECK:STDERR:
  var unused a_ptr: __fn_ptr(form(val i64)) ->? form(var i32) = A;
}

// --- fail_return_type_mismatch.carbon
library "[[@TEST_NAME]]";

fn A(x: i32) -> i32;

fn F() {
  // CHECK:STDERR: fail_return_type_mismatch.carbon:[[@LINE+7]]:65: error: type `i64` in function pointer type differs from type `i32` in function signature [FunctionPtrDiffersType]
  // CHECK:STDERR:   var unused a_ptr: __fn_ptr(form(val i32)) ->? form(var i64) = A;
  // CHECK:STDERR:                                                                 ^
  // CHECK:STDERR: fail_return_type_mismatch.carbon:[[@LINE-6]]:17: note: declared in function signature here [FunctionPtrDiffersPrevious]
  // CHECK:STDERR: fn A(x: i32) -> i32;
  // CHECK:STDERR:                 ^~~
  // CHECK:STDERR:
  var unused a_ptr: __fn_ptr(form(val i32)) ->? form(var i64) = A;
}

// --- fail_implicit_param.carbon
library "[[@TEST_NAME]]";

fn A[x: i32]() -> i32;

fn F() {
  // CHECK:STDERR: fail_implicit_param.carbon:[[@LINE+7]]:65: error: can't form pointer to function with compile-time parameter [FunctionPtrWithCompileTimeParam]
  // CHECK:STDERR:   var unused a_ptr: __fn_ptr(form(val i32)) ->? form(var i32) = A;
  // CHECK:STDERR:                                                                 ^
  // CHECK:STDERR: fail_implicit_param.carbon:[[@LINE-6]]:6: note: declared here [FunctionPtrWithCompileTimeParamPattern]
  // CHECK:STDERR: fn A[x: i32]() -> i32;
  // CHECK:STDERR:      ^~~~~~
  // CHECK:STDERR:
  var unused a_ptr: __fn_ptr(form(val i32)) ->? form(var i32) = A;
}

// --- fail_explicit_generic_param.carbon
library "[[@TEST_NAME]]";

fn A(generic x: i32) -> i32;

fn F() {
  // CHECK:STDERR: fail_explicit_generic_param.carbon:[[@LINE+7]]:65: error: can't form pointer to function with compile-time parameter [FunctionPtrWithCompileTimeParam]
  // CHECK:STDERR:   var unused a_ptr: __fn_ptr(form(val i32)) ->? form(var i32) = A;
  // CHECK:STDERR:                                                                 ^
  // CHECK:STDERR: fail_explicit_generic_param.carbon:[[@LINE-6]]:14: note: declared here [FunctionPtrWithCompileTimeParamPattern]
  // CHECK:STDERR: fn A(generic x: i32) -> i32;
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  var unused a_ptr: __fn_ptr(form(val i32)) ->? form(var i32) = A;
}

// CHECK:STDOUT: --- fn_ptr_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %.c2a: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %tuple.type.3ab: type = tuple_type (Core.Form) [concrete]
// CHECK:STDOUT:   %tuple.89e: %tuple.type.3ab = tuple_value (%.c2a) [concrete]
// CHECK:STDOUT:   %.600: type = fn_ptr_type (%.c2a), %.795f [concrete]
// CHECK:STDOUT:   %pattern_type.3c5: type = pattern_type %.600 [concrete]
// CHECK:STDOUT:   %a_ptr.patt: %pattern_type.3c5 = ref_binding_pattern a_ptr [concrete]
// CHECK:STDOUT:   %a_ptr.var_patt: %pattern_type.3c5 = var_pattern %a_ptr.patt [concrete]
// CHECK:STDOUT:   %.4e3: %.600 = addr_of_fn @A [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Op.type: type = fn_type @Op [concrete]
// CHECK:STDOUT:   %Op: %Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1e9: <witness> = custom_witness (%Op), @Copy [concrete]
// CHECK:STDOUT:   %Copy.facet.c85: %Copy.type = facet_value %.600, (%custom_witness.1e9) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.712: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c85) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %Copy.WithSelf.Op.type.712, %Copy.facet.c85 [concrete]
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %.4e3, %Op [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %result.patt: %pattern_type.6b6 = ref_binding_pattern result [concrete]
// CHECK:STDOUT:   %result.var_patt: %pattern_type.6b6 = var_pattern %result.patt [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %__fn_ptr_thunk.type: type = fn_type @__fn_ptr_thunk [concrete]
// CHECK:STDOUT:   %__fn_ptr_thunk: %__fn_ptr_thunk.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a2a: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1aa, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a2a) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.8eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.b7a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.8eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc8_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc7 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %__fn_ptr_thunk.decl: %__fn_ptr_thunk.type = fn_decl @__fn_ptr_thunk [concrete = constants.%__fn_ptr_thunk] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ptr.var: ref %.600 = var_storage %a_ptr.var_patt
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:   %.loc7_58: %.600 = addr_of_fn @A [concrete = constants.%.4e3]
// CHECK:STDOUT:   %.loc7_3: %.600 = converted %A.ref, %.loc7_58 [concrete = constants.%.4e3]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %impl.elem0.loc7: %.979 = impl_witness_access constants.%custom_witness.1e9, element0 [concrete = constants.%Op]
// CHECK:STDOUT:   %bound_method.loc7: <bound method> = bound_method %.loc7_3, %impl.elem0.loc7 [concrete = constants.%Op.bound]
// CHECK:STDOUT:   %Op.call: init %.600 = call %bound_method.loc7(%.loc7_3) [concrete = constants.%.4e3]
// CHECK:STDOUT:   assign %a_ptr.var, %Op.call
// CHECK:STDOUT:   %.loc7_14.1: type = splice_block %.loc7_14.2 [concrete = constants.%.600] {
// CHECK:STDOUT:     %i32.loc7_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_28: Core.Form = value_form %i32.loc7_32 [concrete = constants.%.c2a]
// CHECK:STDOUT:     %.loc7_36: %tuple.type.3ab = tuple_literal (%.loc7_28) [concrete = constants.%tuple.89e]
// CHECK:STDOUT:     %i32.loc7_51: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_47: Core.Form = init_form %i32.loc7_51 [concrete = constants.%.795f]
// CHECK:STDOUT:     %.loc7_14.2: type = fn_ptr_type (%.loc7_28), %.loc7_47 [concrete = constants.%.600]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr: ref %.600 = wrapper_binding a_ptr, %a_ptr.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ptr.patt: %pattern_type.3c5 = ref_binding_pattern a_ptr [concrete = constants.%a_ptr.patt]
// CHECK:STDOUT:     %a_ptr.var_patt: %pattern_type.3c5 = var_pattern %a_ptr.patt [concrete = constants.%a_ptr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %result.var: ref %i32 = var_storage %result.var_patt
// CHECK:STDOUT:   %a_ptr.ref: ref %.600 = name_ref a_ptr, %a_ptr
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc8_28: %.600 = acquire_value %a_ptr.ref
// CHECK:STDOUT:   %impl.elem0.loc8: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc8_34.1: <bound method> = bound_method %int_0, %impl.elem0.loc8 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc8, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc8_34.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc8_34.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc8_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc8_34.2: %i32 = converted %int_0, %.loc8_34.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %__fn_ptr_thunk.call: init %i32 = call generated.%__fn_ptr_thunk.decl(%.loc8_28, %.loc8_34.2)
// CHECK:STDOUT:   assign %result.var, %__fn_ptr_thunk.call
// CHECK:STDOUT:   %i32.loc8: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %result: ref %i32 = wrapper_binding result, %result.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %result.patt: %pattern_type.6b6 = ref_binding_pattern result [concrete = constants.%result.patt]
// CHECK:STDOUT:     %result.var_patt: %pattern_type.6b6 = var_pattern %result.patt [concrete = constants.%result.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.bound.loc8: <bound method> = bound_method %result.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc8: init %empty_tuple.type = call %Destroy.Op.bound.loc8(%result.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc7: <bound method> = bound_method %a_ptr.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc7: init %empty_tuple.type = call %Destroy.Op.bound.loc7(%a_ptr.var)
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
// CHECK:STDOUT: fn @Destroy.Op.loc7(%self.param: ref %.600) = "no_op";
// CHECK:STDOUT:

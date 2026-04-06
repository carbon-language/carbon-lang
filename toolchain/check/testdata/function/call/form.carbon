// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/form.carbon

// --- ref_form_param.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x:? form(ref i32));
//@dump-sem-ir-end

fn G() {
  var y: i32 = 0;
  //@dump-sem-ir-begin
  F(ref y);
  //@dump-sem-ir-end
}

// --- var_form_param.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x:? form(var i32));
//@dump-sem-ir-end

fn G() {
  //@dump-sem-ir-begin
  F(0);
  //@dump-sem-ir-end
}

// --- val_form_param.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x:? form(val i32));
//@dump-sem-ir-end

fn G() {
  //@dump-sem-ir-begin
  F(0);
  //@dump-sem-ir-end
}

// --- type_component_needs_conversion.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(x:? form(ref ()));
//@dump-sem-ir-end

// --- fail_param_form_not_constant.carbon
library "[[@TEST_NAME]]";

fn F() -> Core.Form();
// CHECK:STDERR: fail_param_form_not_constant.carbon:[[@LINE+4]]:10: error: cannot evaluate form expression [FormExprEvaluationFailure]
// CHECK:STDERR: fn G(x:? F());
// CHECK:STDERR:          ^~~
// CHECK:STDERR:
fn G(x:? F());

// --- fail_form_param_not_form.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_form_param_not_form.carbon:[[@LINE+7]]:10: error: cannot implicitly convert expression of type `type` to `Core.Form` [ConversionFailure]
// CHECK:STDERR: fn F(x:? i32);
// CHECK:STDERR:          ^~~
// CHECK:STDERR: fail_form_param_not_form.carbon:[[@LINE+4]]:10: note: type `type` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F(x:? i32);
// CHECK:STDERR:          ^~~
// CHECK:STDERR:
fn F(x:? i32);
// CHECK:STDERR: fail_form_param_not_form.carbon:[[@LINE+8]]:6: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn G(ref x:? i32);
// CHECK:STDERR:      ^~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_form_param_not_form.carbon:[[@LINE+4]]:6: error: expected `:` binding after `ref` [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR: fn G(ref x:? i32);
// CHECK:STDERR:      ^~~
// CHECK:STDERR:
fn G(ref x:? i32);

// --- fail_missing_ref_tag.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(ref i32));

fn G() {
  var y: i32 = 0;
  // CHECK:STDERR: fail_missing_ref_tag.carbon:[[@LINE+7]]:5: error: argument to `ref` parameter not marked with `ref` [RefParamNoRefTag]
  // CHECK:STDERR:   F(y);
  // CHECK:STDERR:     ^
  // CHECK:STDERR: fail_missing_ref_tag.carbon:[[@LINE-7]]:6: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F(x:? form(ref i32));
  // CHECK:STDERR:      ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(y);
}

// --- fail_todo_form_generic_param.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_form_generic_param.carbon:[[@LINE+4]]:26: error: semantics TODO: `Support symbolic form bindings` [SemanticsTodo]
// CHECK:STDERR: fn F(Form:! Core.Form(), x:? Form);
// CHECK:STDERR:                          ^~~~~~~~
// CHECK:STDERR:
fn F(Form:! Core.Form(), x:? Form);

fn G() {
  var x: i32;
  F(form(var i32), x);
}

// --- fail_todo_local_init_form_binding.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR: fail_todo_local_init_form_binding.carbon:[[@LINE+12]]:7: error: semantics TODO: `Support local initializing forms` [SemanticsTodo]
  // CHECK:STDERR:   let x:? form(var i32) = 0;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_local_init_form_binding.carbon:[[@LINE+8]]:27: error: cannot bind durable reference to non-reference value of type `i32` [ConversionFailureNonRefToRef]
  // CHECK:STDERR:   let x:? form(var i32) = 0;
  // CHECK:STDERR:                           ^
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_local_init_form_binding.carbon:[[@LINE+4]]:7: warning: binding `x` unused [UnusedBinding]
  // CHECK:STDERR:   let x:? form(var i32) = 0;
  // CHECK:STDERR:       ^~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  let x:? form(var i32) = 0;
}

// --- fail_todo_local_symbolic_form_binding.carbon
library "[[@TEST_NAME]]";

// TODO: this code should fail for a different reason, but right now we don't
// have a way to write correct code that reaches the TODO we're exercising here.
fn F(Form:! Core.Form()) {
  // CHECK:STDERR: fail_todo_local_symbolic_form_binding.carbon:[[@LINE+4]]:7: error: semantics TODO: `Support symbolic form bindings` [SemanticsTodo]
  // CHECK:STDERR:   let y:? Form = 0;
  // CHECK:STDERR:       ^~~~~~~~
  // CHECK:STDERR:
  let y:? Form = 0;
}

// --- ref_return_form.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F() ->? form(ref i32);
//@dump-sem-ir-end

fn G() {
  //@dump-sem-ir-begin
  let ref _: i32 = F();
  //@dump-sem-ir-end
}

// --- var_return_form.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F() ->? form(var i32);
//@dump-sem-ir-end

fn G() {
  //@dump-sem-ir-begin
  let var _: i32 = F();
  //@dump-sem-ir-end
}

// --- val_return_form.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F() ->? form(val i32);
//@dump-sem-ir-end

fn G() {
  //@dump-sem-ir-begin
  let _: i32 = F();
  //@dump-sem-ir-end
}

// --- fail_return_form_not_form.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+7]]:12: error: cannot implicitly convert expression of type `type` to `Core.Form` [ConversionFailure]
// CHECK:STDERR: fn F() ->? i32;
// CHECK:STDERR:            ^~~
// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+4]]:12: note: type `type` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F() ->? i32;
// CHECK:STDERR:            ^~~
// CHECK:STDERR:
fn F() ->? i32;
// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+11]]:12: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
// CHECK:STDERR: fn F() ->? ref i32;
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+7]]:12: error: cannot implicitly convert expression of type `type` to `Core.Form` [ConversionFailure]
// CHECK:STDERR: fn F() ->? ref i32;
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+4]]:12: note: type `type` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
// CHECK:STDERR: fn F() ->? ref i32;
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR:
fn F() ->? ref i32;

// -- fail_todo_form_generic_return.carbon

// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+11]]:26: error: semantics TODO: `Support symbolic return forms` [SemanticsTodo]
// CHECK:STDERR: fn F(Form:! Core.Form()) ->? Form;
// CHECK:STDERR:                          ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+7]]:1: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
// CHECK:STDERR: fn F(Form:! Core.Form()) ->? Form;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE-23]]:1: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
// CHECK:STDERR: fn F() ->? i32;
// CHECK:STDERR: ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(Form:! Core.Form()) ->? Form;

// CHECK:STDOUT: --- ref_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.1da: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: %pattern_type.7ce = form_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = at_binding_pattern x, %.loc4_7 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.1da] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %i32 [concrete = constants.%.1da]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %i32 = form_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: ref %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %y.ref: ref %i32 = name_ref y, %y
// CHECK:STDOUT:   %.loc10: %i32 = ref_tag %y.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc10)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %.eea: ref %i32 = temporary invalid, %int_0.6a9 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc4_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.eea, %Destroy.Op.651ba6.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: %pattern_type.7ce = form_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = at_binding_pattern x, %.loc4_7 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.ff5] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %i32 = form_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: ref %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc4_7.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc4_7.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_7.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_7.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_7.2: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc4_7.3: ref %i32 = temporary %.loc4_7.1, %.loc4_7.2 [concrete = constants.%.eea]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc4_7.3)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.eea)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc4_7.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc4_7.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.754: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: %pattern_type.7ce = form_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = at_binding_pattern x, %.loc4_7 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.754] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = value_form %i32 [concrete = constants.%.754]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = form_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_5.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_5.2: %i32 = converted %int_0, %.loc9_5.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc9_5.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_component_needs_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.9f9: Core.Form = ref_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_7: %pattern_type = form_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type = at_binding_pattern x, %.loc4_7 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.9f9] {
// CHECK:STDOUT:       %.loc4_20.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_20.2: type = converted %.loc4_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %.loc4_20.2 [concrete = constants.%.9f9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %empty_tuple.type = form_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: ref %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_return_form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.1da: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4: Core.Form = ref_form %i32 [concrete = constants.%.1da]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> ref %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.7ce = ref_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: ref %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: ref %i32 = ref_binding _, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_return_form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc9_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, constants.%i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.7ce = ref_binding_pattern _ [concrete]
// CHECK:STDOUT:     %_.var_patt: %pattern_type.7ce = var_pattern %_.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.var: ref %i32 = var %_.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   assign %_.var, %F.call
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: ref %i32 = ref_binding _, %_.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %_.var, constants.%Destroy.Op.651ba6.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%_.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_7.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_7.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_return_form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.754: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4: Core.Form = value_form %i32 [concrete = constants.%.754]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> val %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.7ce = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: %i32 = value_binding _, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

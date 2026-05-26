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
  // CHECK:STDERR: fail_todo_local_init_form_binding.carbon:[[@LINE+4]]:7: error: semantics TODO: `support local form bindings` [SemanticsTodo]
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

// --- form_generic_return.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(Form:! Core.Form()) ->? Form;
//@dump-sem-ir-end

// CHECK:STDOUT: --- ref_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.a8d: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c6b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.c6b = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.a8d] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %i32 [concrete = constants.%.a8d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %i32 = ref_binding x, %x.param
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
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.155: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc4_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c6b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.c6b = var_pattern %x.param_patt [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.c6b = at_binding_pattern x, %x.var_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.70c] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %i32 = ref_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: ref %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc4_7.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_7.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_7.2(%int_0) [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc4: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %x.var: ref %i32 = var @F.%x.var_patt
// CHECK:STDOUT:   assign %x.var, %.loc4
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%x.var)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%x.var)
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
// CHECK:STDOUT:   %.688: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.155: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.c6b = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.c6b = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.688] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = value_form %i32 [concrete = constants.%.688]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_5.2(%int_0) [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc9_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.155]
// CHECK:STDOUT:   %.loc9_5.2: %i32 = converted %int_0, %.loc9_5.1 [concrete = constants.%int_0.155]
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
// CHECK:STDOUT:     %x.param_patt: %pattern_type = ref_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.9f9] {
// CHECK:STDOUT:       %.loc4_20.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_20.2: type = converted %.loc4_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %.loc4_20.2 [concrete = constants.%.9f9]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %empty_tuple.type = ref_binding x, %x.param
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
// CHECK:STDOUT:   %.a8d: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.c6b = ref_return_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %.loc4_17.1, constants.%i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = ref_form %i32 [concrete = constants.%.a8d]
// CHECK:STDOUT:     %.loc4_17.3: ref %i32 = ref_return
// CHECK:STDOUT:     %return: ref %i32 = return_slot %.loc4_17.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> ref %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.c6b = ref_binding_pattern _ [concrete]
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
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc9_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, constants.%i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4: Core.Form = init_form %i32 [concrete = constants.%.70c]
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
// CHECK:STDOUT:     %_.patt: %pattern_type.c6b = ref_binding_pattern _ [concrete]
// CHECK:STDOUT:     %_.var_patt: %pattern_type.c6b = var_pattern %_.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.var: ref %i32 = var %_.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   assign %_.var, %F.call
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: ref %i32 = ref_binding _, %_.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %_.var, constants.%Destroy.Op.1dc86d.2
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
// CHECK:STDOUT:   %.688: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.c6b = value_return_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %.loc4_17.1, constants.%i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = value_form %i32 [concrete = constants.%.688]
// CHECK:STDOUT:     %.loc4_17.3: %i32 = value_return
// CHECK:STDOUT:     %return: %i32 = return_slot %.loc4_17.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> val %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.c6b = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: %i32 = value_binding _, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- form_generic_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Form.type: type = fn_type @Form [concrete]
// CHECK:STDOUT:   %Form.cd3: %Form.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.13f: type = pattern_type Core.Form [concrete]
// CHECK:STDOUT:   %Form.ac3: Core.Form = symbolic_binding Form, 0 [symbolic]
// CHECK:STDOUT:   %.344: Core.Form = splice_inst @F.%.loc4_30.1 [template]
// CHECK:STDOUT:   %.82f: type = type_component_of %.344 [template]
// CHECK:STDOUT:   %pattern_type.ec7: type = pattern_type %.82f [template]
// CHECK:STDOUT:   %.f74: %.82f = splice_inst @F.%.loc4_26.2 [template]
// CHECK:STDOUT:   %.33d: <instruction> = refine_form_action %Form.ac3 [template]
// CHECK:STDOUT:   %.80a: Core.Form = splice_inst %.33d [template]
// CHECK:STDOUT:   %.af3: <instruction> = out_form_param_pattern_action %.80a [template]
// CHECK:STDOUT:   %.465: type = type_component_of %.80a [template]
// CHECK:STDOUT:   %.e7f: %.465 = splice_inst %.af3 [template]
// CHECK:STDOUT:   %pattern_type.a51: type = pattern_type %.465 [template]
// CHECK:STDOUT:   %.cab: <instruction> = callee_pattern_match_action %.e7f, call_param0 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Form = %Core.Form
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Form: %Form.type = import_ref Core//prelude/parts/form, Form, loaded [concrete = constants.%Form.cd3]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %Form.patt: %pattern_type.13f = symbolic_binding_pattern Form, 0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type (%pattern_type.ec7) = return_slot_pattern %.loc4_26.6, constants.%.82f [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Form.ref.loc4_30: Core.Form = name_ref Form, %Form.loc4_10.2 [symbolic = %Form.loc4_10.1 (constants.%Form.ac3)]
// CHECK:STDOUT:     %.loc4_30.3: Core.Form = splice_inst %.loc4_30.1 [template = %.loc4_30.2 (constants.%.344)]
// CHECK:STDOUT:     %.loc4_26.6: @F.%.loc4_26.3 (%.82f) = splice_inst %.loc4_26.2 [template = %.loc4_26.4 (constants.%.f74)]
// CHECK:STDOUT:     %.loc4_23.1: type = splice_block %.loc4_23.3 [concrete = Core.Form] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Form.ref.loc4_17: %Form.type = name_ref Form, imports.%Core.Form [concrete = constants.%Form.cd3]
// CHECK:STDOUT:       %Form.call: init type = call %Form.ref.loc4_17() [concrete = Core.Form]
// CHECK:STDOUT:       %.loc4_23.2: type = value_of_initializer %Form.call [concrete = Core.Form]
// CHECK:STDOUT:       %.loc4_23.3: type = converted %Form.call, %.loc4_23.2 [concrete = Core.Form]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Form.loc4_10.2: Core.Form = symbolic_binding Form, 0 [symbolic = %Form.loc4_10.1 (constants.%Form.ac3)]
// CHECK:STDOUT:     %.loc4_26.1: @F.%.loc4_26.3 (%.82f) = splice_inst %.loc4_26.5
// CHECK:STDOUT:     %return: @F.%.loc4_26.3 (%.82f) = return_slot %.loc4_26.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%Form.loc4_10.2: Core.Form) {
// CHECK:STDOUT:   %Form.loc4_10.1: Core.Form = symbolic_binding Form, 0 [symbolic = %Form.loc4_10.1 (constants.%Form.ac3)]
// CHECK:STDOUT:   %.loc4_30.1: <instruction> = refine_form_action %Form.ref.loc4_30 [template]
// CHECK:STDOUT:   %.loc4_30.2: Core.Form = splice_inst %.loc4_30.1 [template = %.loc4_30.2 (constants.%.344)]
// CHECK:STDOUT:   %.loc4_26.2: <instruction> = out_form_param_pattern_action %.loc4_30.3 [template]
// CHECK:STDOUT:   %.loc4_26.3: type = type_component_of %.loc4_30.2 [template = %.loc4_26.3 (constants.%.82f)]
// CHECK:STDOUT:   %.loc4_26.4: @F.%.loc4_26.3 (%.82f) = splice_inst %.loc4_26.2 [template = %.loc4_26.4 (constants.%.f74)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc4_26.3 [template = %pattern_type (constants.%pattern_type.ec7)]
// CHECK:STDOUT:   %.loc4_26.5: <instruction> = callee_pattern_match_action %.loc4_26.6, call_param0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %.loc4_26.1:? @F.%.loc4_30.1 (@F.%.loc4_30.1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Form.ac3) {
// CHECK:STDOUT:   %Form.loc4_10.1 => constants.%Form.ac3
// CHECK:STDOUT:   %.loc4_30.1 => constants.%.33d
// CHECK:STDOUT:   %.loc4_30.2 => constants.%.80a
// CHECK:STDOUT:   %.loc4_26.2 => constants.%.af3
// CHECK:STDOUT:   %.loc4_26.3 => constants.%.465
// CHECK:STDOUT:   %.loc4_26.4 => constants.%.e7f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a51
// CHECK:STDOUT:   %.loc4_26.5 => constants.%.cab
// CHECK:STDOUT: }
// CHECK:STDOUT:

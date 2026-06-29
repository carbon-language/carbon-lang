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

fn F() -> Core.Form;
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

// --- form_generic_param.carbon
library "[[@TEST_NAME]]";

fn F(Fm:! Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  F(form(var i32), x);
}

// --- fail_todo_composite_generic_form.carbon
library "[[@TEST_NAME]]";

fn F(Fm:! Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR: fail_todo_composite_generic_form.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `(Core.Form, Core.Form)` to `Core.Form` [ConversionFailure]
  // CHECK:STDERR:   F((form(var i32), form(ref i32)), (x, ref x));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_composite_generic_form.carbon:[[@LINE+7]]:3: note: type `(Core.Form, Core.Form)` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   F((form(var i32), form(ref i32)), (x, ref x));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_composite_generic_form.carbon:[[@LINE-10]]:6: note: initializing generic parameter `Fm` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(Fm:! Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:      ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  F((form(var i32), form(ref i32)), (x, ref x));
}

// --- fail_todo_deduce_generic_form.carbon
library "[[@TEST_NAME]]";

fn F[Fm:! Core.Form](v:? Fm);

fn G() {
  var v: i32 = 0;
  // CHECK:STDERR: fail_todo_deduce_generic_form.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `Fm` [DeductionIncomplete]
  // CHECK:STDERR:   F(ref v);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_deduce_generic_form.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[Fm:! Core.Form](v:? Fm);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(ref v);
}

// --- fail_form_generic_arg_not_form.carbon
library "[[@TEST_NAME]]";

fn F(Fm:! Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR: fail_form_generic_arg_not_form.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `type` to `Core.Form` [ConversionFailure]
  // CHECK:STDERR:   F(i32, x);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_form_generic_arg_not_form.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   F(i32, x);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_form_generic_arg_not_form.carbon:[[@LINE-10]]:6: note: initializing generic parameter `Fm` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(Fm:! Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:      ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(i32, x);
}

// --- fail_form_generic_missing_ref_tag.carbon
library "[[@TEST_NAME]]";

fn F(Fm:! Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR: fail_form_generic_missing_ref_tag.carbon:[[@LINE+8]]:20: error: argument to `ref` parameter not marked with `ref` [RefParamNoRefTag]
  // CHECK:STDERR:   F(form(ref i32), x);
  // CHECK:STDERR:                    ^
  // CHECK:STDERR: fail_form_generic_missing_ref_tag.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fail_form_generic_missing_ref_tag.carbon:[[@LINE-8]]:29: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F(Fm:! Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:                             ^~~~~~
  // CHECK:STDERR:
  F(form(ref i32), x);
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
fn F(Form:! Core.Form) {
  // CHECK:STDERR: fail_todo_local_symbolic_form_binding.carbon:[[@LINE+4]]:7: error: semantics TODO: `support local form bindings` [SemanticsTodo]
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
// CHECK:STDERR: fail_return_form_not_form.carbon:[[@LINE+4]]:12: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
// CHECK:STDERR: fn F() ->? ref i32;
// CHECK:STDERR:            ^~~~~~~
// CHECK:STDERR:
fn F() ->? ref i32;

// --- form_generic_return.carbon
library "[[@TEST_NAME]]";
//@include-in-dumps

fn F(Fm:! Core.Form, v:? Fm) ->? Fm {
  // TODO: This should fail because `Fm` might be `form(var T)` for a
  // non-copyable `T`.
  return v;
}

fn G() {
  var v: i32 = 0;

  var unused x: i32 = F(form(var i32), v);
  let unused ref y: i32 = F(form(ref i32), ref v);
  let unused z: i32 = F(form(val i32), v);
}

// CHECK:STDOUT: --- ref_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.512: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.512] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %i32 [concrete = constants.%.512]
// CHECK:STDOUT:       %.loc4_15.3: type = type_component_of %.loc4_15.2 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %i32 = wrapper_binding x, %x.param
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
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%y.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.6b6 = var_pattern %x.param_patt [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.var_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc4_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6b6 = var_pattern %x.param_patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.var_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.795] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:       %.loc4_15.3: type = type_component_of %.loc4_15.2 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: ref %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc4_7.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_7.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_7.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc4: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %x.var: ref %i32 = var @F.%x.var_patt
// CHECK:STDOUT:   assign %x.var, %.loc4
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%x.var)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %x.var, constants.%Destroy.Op.1a2547.2
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
// CHECK:STDOUT:   %.c2a: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.c2a] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = value_form %i32 [concrete = constants.%.c2a]
// CHECK:STDOUT:       %.loc4_15.3: type = type_component_of %.loc4_15.2 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%x.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_5.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc9_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc9_5.2: %i32 = converted %int_0, %.loc9_5.1 [concrete = constants.%int_0.3c0]
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
// CHECK:STDOUT:   %x.param_patt: %pattern_type = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = at_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %empty_tuple.type = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.9f9] {
// CHECK:STDOUT:       %.loc4_20.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc4_20.2: type = converted %.loc4_20.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %.loc4_20.2 [concrete = constants.%.9f9]
// CHECK:STDOUT:       %.loc4_15.3: type = type_component_of %.loc4_15.2 [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref %empty_tuple.type = wrapper_binding x, %x.param
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
// CHECK:STDOUT:   %.512: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ae0: %pattern_type.6b6 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.6b6 = return_slot_pattern %.ae0, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.6b6 = ref_binding_pattern _ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.6b6 = ref_return_pattern [concrete = constants.%.ae0]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = ref_form %i32 [concrete = constants.%.512]
// CHECK:STDOUT:     %.loc4_17.3: type = type_component_of %.loc4_17.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.4: ref %i32 = ref_return
// CHECK:STDOUT:     %return: ref %i32 = return_slot %.loc4_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> ref %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.6b6 = ref_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: ref %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: ref %i32 = wrapper_binding _, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_return_form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.6b6 = ref_binding_pattern _ [concrete]
// CHECK:STDOUT:   %_.var_patt: %pattern_type.6b6 = var_pattern %_.patt [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc9_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %.loc4_17.2 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.1: Core.Form = init_form %i32 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc4_17.2: type = type_component_of %.loc4_17.1 [concrete = constants.%i32]
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
// CHECK:STDOUT:     %_.patt: %pattern_type.6b6 = ref_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:     %_.var_patt: %pattern_type.6b6 = var_pattern %_.patt [concrete = constants.%_.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %_.var: ref %i32 = var %_.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   assign %_.var, %F.call
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: ref %i32 = wrapper_binding _, %_.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %_.var, constants.%Destroy.Op.1a2547.2
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
// CHECK:STDOUT:   %.c2a: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.1a5: %pattern_type.6b6 = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.6b6 = return_slot_pattern %.1a5, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.6b6 = value_binding_pattern _ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.6b6 = value_return_pattern [concrete = constants.%.1a5]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = value_form %i32 [concrete = constants.%.c2a]
// CHECK:STDOUT:     %.loc4_17.3: type = type_component_of %.loc4_17.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.4: %i32 = value_return
// CHECK:STDOUT:     %return: %i32 = return_slot %.loc4_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> val %i32;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.6b6 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: %i32 = wrapper_binding _, %F.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- form_generic_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.13f: type = pattern_type Core.Form [concrete]
// CHECK:STDOUT:   %Fm.patt: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic]
// CHECK:STDOUT:   %Fm: Core.Form = symbolic_binding Fm, 0 [symbolic]
// CHECK:STDOUT:   %.42f: type = type_component_of %Fm [symbolic]
// CHECK:STDOUT:   %pattern_type.177: type = pattern_type %.42f [symbolic]
// CHECK:STDOUT:   %.f15: %pattern_type.177 = splice_inst @F.%.loc4_23.3 [template]
// CHECK:STDOUT:   %v.patt.0b3: %pattern_type.177 = at_binding_pattern v, %.f15 [template]
// CHECK:STDOUT:   %.677: %pattern_type.177 = splice_inst @F.%.loc4_30.3 [template]
// CHECK:STDOUT:   %return.patt.113: %pattern_type.177 = return_slot_pattern %.677, %.42f [template]
// CHECK:STDOUT:   %.cfa: <instruction> = form_param_pattern_action %Fm, v [template]
// CHECK:STDOUT:   %.26b: %pattern_type.177 = splice_inst %.cfa [template]
// CHECK:STDOUT:   %v.patt.cda: %pattern_type.177 = at_binding_pattern v, %.26b [template]
// CHECK:STDOUT:   %.94f: <instruction> = out_form_param_pattern_action %Fm [template]
// CHECK:STDOUT:   %.1ac: %pattern_type.177 = splice_inst %.94f [template]
// CHECK:STDOUT:   %return.patt.333: %pattern_type.177 = return_slot_pattern %.1ac, %.42f [template]
// CHECK:STDOUT:   %.095: <instruction> = callee_pattern_match_action %.f15, call_param0 [template]
// CHECK:STDOUT:   %.efc: <instruction> = callee_pattern_match_action %.677, call_param1 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c9b: <witness> = require_complete_type %.42f [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %v.patt.113: %pattern_type.6b6 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.68d: %pattern_type.6b6 = var_pattern %v.patt.113 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a4c7.1: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.6b6 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.6b6 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %v.param_patt.e9ad17.1: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %v.var_patt.13561f.1: %pattern_type.6b6 = var_pattern %v.param_patt.e9ad17.1 [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.eb8: %pattern_type.6b6 = splice_block %v.var_patt.13561f.2 [concrete = %v.var_patt.13561f.1] {
// CHECK:STDOUT:       %v.param_patt.e9ad17.3: %pattern_type.6b6 = ref_param_pattern [concrete = %v.param_patt.e9ad17.1]
// CHECK:STDOUT:       %v.var_patt.13561f.2: %pattern_type.6b6 = var_pattern %v.param_patt.e9ad17.3 [concrete = %v.var_patt.13561f.1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.b2b: %pattern_type.6b6 = at_binding_pattern v, %v.var_patt.13561f.1 [concrete]
// CHECK:STDOUT:   %inst.out_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %return.param_patt.a9a4c7.2: %pattern_type.6b6 = out_param_pattern [concrete = %return.param_patt.a9a4c7.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a4c7.1, %i32 [concrete]
// CHECK:STDOUT:   %F.specific_fn.8ea: <specific function> = specific_function %F, @F(%.795) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.6b6 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %.512: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %inst.ref_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %v.param_patt.e9ad17.2: %pattern_type.6b6 = ref_param_pattern [concrete = %v.param_patt.e9ad17.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.5c0: %pattern_type.6b6 = at_binding_pattern v, %v.param_patt.e9ad17.1 [concrete]
// CHECK:STDOUT:   %.ae0167.1: %pattern_type.6b6 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %inst.ref_return_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ae0167.2: %pattern_type.6b6 = ref_return_pattern [concrete = %.ae0167.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.c62: %pattern_type.6b6 = return_slot_pattern %.ae0167.1, %i32 [concrete]
// CHECK:STDOUT:   %F.specific_fn.0c3: <specific function> = specific_function %F, @F(%.512) [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.6b6 = value_binding_pattern z [concrete]
// CHECK:STDOUT:   %.c2a: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %v.param_patt.733534.1: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %inst.value_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %v.param_patt.733534.2: %pattern_type.6b6 = value_param_pattern [concrete = %v.param_patt.733534.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.971: %pattern_type.6b6 = at_binding_pattern v, %v.param_patt.733534.1 [concrete]
// CHECK:STDOUT:   %.1a588f.1: %pattern_type.6b6 = value_return_pattern [concrete]
// CHECK:STDOUT:   %inst.value_return_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.1a588f.2: %pattern_type.6b6 = value_return_pattern [concrete = %.1a588f.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.45a: %pattern_type.6b6 = return_slot_pattern %.1a588f.1, %i32 [concrete]
// CHECK:STDOUT:   %F.specific_fn.2f1: <specific function> = specific_function %F, @F(%.c2a) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d07: %Destroy.type = facet_value %i32, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.47a: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.d07) [concrete]
// CHECK:STDOUT:   %.22e: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.47a, %Destroy.facet.d07 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Form = %Core.Form
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Form: type = import_ref Core//prelude/parts/form, Form, loaded [concrete = Core.Form]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.c64: %Destroy.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.ae8]
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
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
// CHECK:STDOUT:     %Fm.patt.loc4_8.1: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic = %Fm.patt.loc4_8.2 (constants.%Fm.patt)]
// CHECK:STDOUT:     %.loc4_23.2: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_23.3 [template = %.loc4_23.4 (constants.%.f15)]
// CHECK:STDOUT:     %v.patt.loc4_23.1: @F.%pattern_type (%pattern_type.177) = at_binding_pattern v, %.loc4_23.2 [template = %v.patt.loc4_23.2 (constants.%v.patt.0b3)]
// CHECK:STDOUT:     %.loc4_30.2: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_30.3 [template = %.loc4_30.4 (constants.%.677)]
// CHECK:STDOUT:     %return.patt.loc4_30.1: @F.%pattern_type (%pattern_type.177) = return_slot_pattern %.loc4_30.2, %.loc4_34 [template = %return.patt.loc4_30.2 (constants.%return.patt.113)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Fm.ref.loc4_34: Core.Form = name_ref Fm, %Fm.loc4_8.2 [symbolic = %Fm.loc4_8.1 (constants.%Fm)]
// CHECK:STDOUT:     %.loc4_34: type = type_component_of %Fm.ref.loc4_34 [symbolic = %.loc4_26.1 (constants.%.42f)]
// CHECK:STDOUT:     %.loc4_15: type = splice_block %Form.ref [concrete = Core.Form] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Form.ref: type = name_ref Form, imports.%Core.Form [concrete = Core.Form]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Fm.loc4_8.2: Core.Form = symbolic_binding Fm, 0 [symbolic = %Fm.loc4_8.1 (constants.%Fm)]
// CHECK:STDOUT:     %.loc4_23.1: @F.%.loc4_26.1 (%.42f) = splice_inst %.loc4_23.5
// CHECK:STDOUT:     %.loc4_26.2: Core.Form = splice_block %Fm.ref.loc4_26 [symbolic = %Fm.loc4_8.1 (constants.%Fm)] {
// CHECK:STDOUT:       %Fm.ref.loc4_26: Core.Form = name_ref Fm, %Fm.loc4_8.2 [symbolic = %Fm.loc4_8.1 (constants.%Fm)]
// CHECK:STDOUT:       %.loc4_26.3: type = type_component_of %Fm.ref.loc4_26 [symbolic = %.loc4_26.1 (constants.%.42f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @F.%.loc4_26.1 (%.42f) = wrapper_binding v, %.loc4_23.1
// CHECK:STDOUT:     %.loc4_30.1: @F.%.loc4_26.1 (%.42f) = splice_inst %.loc4_30.5
// CHECK:STDOUT:     %return: @F.%.loc4_26.1 (%.42f) = return_slot %.loc4_30.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%Fm.loc4_8.2: Core.Form) {
// CHECK:STDOUT:   %Fm.patt.loc4_8.2: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic = %Fm.patt.loc4_8.2 (constants.%Fm.patt)]
// CHECK:STDOUT:   %Fm.loc4_8.1: Core.Form = symbolic_binding Fm, 0 [symbolic = %Fm.loc4_8.1 (constants.%Fm)]
// CHECK:STDOUT:   %.loc4_26.1: type = type_component_of %Fm.loc4_8.1 [symbolic = %.loc4_26.1 (constants.%.42f)]
// CHECK:STDOUT:   %.loc4_23.3: <instruction> = form_param_pattern_action %Fm.ref.loc4_26, v [template]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc4_26.1 [symbolic = %pattern_type (constants.%pattern_type.177)]
// CHECK:STDOUT:   %.loc4_23.4: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_23.3 [template = %.loc4_23.4 (constants.%.f15)]
// CHECK:STDOUT:   %v.patt.loc4_23.2: @F.%pattern_type (%pattern_type.177) = at_binding_pattern v, %.loc4_23.4 [template = %v.patt.loc4_23.2 (constants.%v.patt.0b3)]
// CHECK:STDOUT:   %.loc4_30.3: <instruction> = out_form_param_pattern_action %Fm.ref.loc4_34 [template]
// CHECK:STDOUT:   %.loc4_30.4: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_30.3 [template = %.loc4_30.4 (constants.%.677)]
// CHECK:STDOUT:   %return.patt.loc4_30.2: @F.%pattern_type (%pattern_type.177) = return_slot_pattern %.loc4_30.4, %.loc4_26.1 [template = %return.patt.loc4_30.2 (constants.%return.patt.113)]
// CHECK:STDOUT:   %.loc4_23.5: <instruction> = callee_pattern_match_action constants.%.f15, call_param0 [template]
// CHECK:STDOUT:   %.loc4_30.5: <instruction> = callee_pattern_match_action constants.%.677, call_param1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc4_26.1 [symbolic = %require_complete (constants.%require_complete.c9b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%.loc4_23.1: @F.%.loc4_26.1 (%.42f)) -> out %.loc4_30.1:? %Fm {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.ref: @F.%.loc4_26.1 (%.42f) = name_ref v, %v
// CHECK:STDOUT:     return %v.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.6b6 = ref_binding_pattern v [concrete = constants.%v.patt.113]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.6b6 = var_pattern %v.patt [concrete = constants.%v.var_patt.68d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var.loc11: ref %i32 = var %v.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc11_3.1: <bound method> = bound_method %int_0, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_3.2: <bound method> = bound_method %int_0, %specific_fn.loc11 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_3.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc11: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   assign %v.var.loc11, %.loc11
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %v: ref %i32 = wrapper_binding v, %v.var.loc11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.6b6 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %i32 = var %x.var_patt
// CHECK:STDOUT:   %F.ref.loc13: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc13_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc13_30: Core.Form = init_form %i32.loc13_34 [concrete = constants.%.795]
// CHECK:STDOUT:   %v.ref.loc13: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %F.specific_fn.loc13: <specific function> = specific_function %F.ref.loc13, @F(constants.%.795) [concrete = constants.%F.specific_fn.8ea]
// CHECK:STDOUT:   %.loc13_40: %i32 = acquire_value %v.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc13_40.1: <bound method> = bound_method %.loc13_40, %impl.elem0.loc13
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_40.2: <bound method> = bound_method %.loc13_40, %specific_fn.loc13
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc13_40.2(%.loc13_40)
// CHECK:STDOUT:   %v.var.1: ref %i32 = var constants.%v.var_patt.13561f.1
// CHECK:STDOUT:   assign %v.var.1, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %F.call.loc13: init %i32 = call %F.specific_fn.loc13(%v.var.1)
// CHECK:STDOUT:   assign %x.var, %F.call.loc13
// CHECK:STDOUT:   %i32.loc13_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.6b6 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc14: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc14_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_34: Core.Form = ref_form %i32.loc14_38 [concrete = constants.%.512]
// CHECK:STDOUT:   %v.ref.loc14: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc14_44: %i32 = ref_tag %v.ref.loc14
// CHECK:STDOUT:   %F.specific_fn.loc14: <specific function> = specific_function %F.ref.loc14, @F(constants.%.512) [concrete = constants.%F.specific_fn.0c3]
// CHECK:STDOUT:   %F.call.loc14: ref %i32 = call %F.specific_fn.loc14(%v.ref.loc14)
// CHECK:STDOUT:   %i32.loc14_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %y: ref %i32 = wrapper_binding y, %F.call.loc14
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.6b6 = value_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc15: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc15_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc15_30: Core.Form = value_form %i32.loc15_34 [concrete = constants.%.c2a]
// CHECK:STDOUT:   %v.ref.loc15: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %F.specific_fn.loc15: <specific function> = specific_function %F.ref.loc15, @F(constants.%.c2a) [concrete = constants.%F.specific_fn.2f1]
// CHECK:STDOUT:   %.loc15_40: %i32 = acquire_value %v.ref.loc15
// CHECK:STDOUT:   %F.call.loc15: %i32 = call %F.specific_fn.loc15(%.loc15_40)
// CHECK:STDOUT:   %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %z: %i32 = wrapper_binding z, %F.call.loc15
// CHECK:STDOUT:   %Op.ref: %Destroy.assoc_type = name_ref Op, imports.%Core.import_ref.c64 [concrete = constants.%assoc0.ae8]
// CHECK:STDOUT:   %impl.elem0.1: %.22e = impl_witness_access constants.%custom_witness.df9cc1.2, element0 [concrete = constants.%Destroy.Op.1a2547.2]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %v.var.1, %impl.elem0.1
// CHECK:STDOUT:   %Destroy.Op.call.1: init %empty_tuple.type = call %bound_method.1(%v.var.1)
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %x.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%x.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc11: <bound method> = bound_method %v.var.loc11, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc11: init %empty_tuple.type = call %Destroy.Op.bound.loc11(%v.var.loc11)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Fm) {
// CHECK:STDOUT:   %Fm.patt.loc4_8.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_8.1 => constants.%Fm
// CHECK:STDOUT:   %.loc4_26.1 => constants.%.42f
// CHECK:STDOUT:   %.loc4_23.3 => constants.%.cfa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.177
// CHECK:STDOUT:   %.loc4_23.4 => constants.%.26b
// CHECK:STDOUT:   %v.patt.loc4_23.2 => constants.%v.patt.cda
// CHECK:STDOUT:   %.loc4_30.3 => constants.%.94f
// CHECK:STDOUT:   %.loc4_30.4 => constants.%.1ac
// CHECK:STDOUT:   %return.patt.loc4_30.2 => constants.%return.patt.333
// CHECK:STDOUT:   %.loc4_23.5 => constants.%.095
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.efc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.795) {
// CHECK:STDOUT:   %Fm.patt.loc4_8.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_8.1 => constants.%.795
// CHECK:STDOUT:   %.loc4_26.1 => constants.%i32
// CHECK:STDOUT:   %.loc4_23.3 => constants.%inst.splice_block
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %.loc4_23.4 => constants.%v.var_patt.13561f.1
// CHECK:STDOUT:   %v.patt.loc4_23.2 => constants.%v.patt.b2b
// CHECK:STDOUT:   %.loc4_30.3 => constants.%inst.out_param_pattern
// CHECK:STDOUT:   %.loc4_30.4 => constants.%return.param_patt.a9a4c7.1
// CHECK:STDOUT:   %return.patt.loc4_30.2 => constants.%return.patt.e1b
// CHECK:STDOUT:   %.loc4_23.5 => constants.%.095
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.efc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.512) {
// CHECK:STDOUT:   %Fm.patt.loc4_8.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_8.1 => constants.%.512
// CHECK:STDOUT:   %.loc4_26.1 => constants.%i32
// CHECK:STDOUT:   %.loc4_23.3 => constants.%inst.ref_param_pattern
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %.loc4_23.4 => constants.%v.param_patt.e9ad17.1
// CHECK:STDOUT:   %v.patt.loc4_23.2 => constants.%v.patt.5c0
// CHECK:STDOUT:   %.loc4_30.3 => constants.%inst.ref_return_pattern
// CHECK:STDOUT:   %.loc4_30.4 => constants.%.ae0167.1
// CHECK:STDOUT:   %return.patt.loc4_30.2 => constants.%return.patt.c62
// CHECK:STDOUT:   %.loc4_23.5 => constants.%.095
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.efc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.c2a) {
// CHECK:STDOUT:   %Fm.patt.loc4_8.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_8.1 => constants.%.c2a
// CHECK:STDOUT:   %.loc4_26.1 => constants.%i32
// CHECK:STDOUT:   %.loc4_23.3 => constants.%inst.value_param_pattern
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %.loc4_23.4 => constants.%v.param_patt.733534.1
// CHECK:STDOUT:   %v.patt.loc4_23.2 => constants.%v.patt.971
// CHECK:STDOUT:   %.loc4_30.3 => constants.%inst.value_return_pattern
// CHECK:STDOUT:   %.loc4_30.4 => constants.%.1a588f.1
// CHECK:STDOUT:   %return.patt.loc4_30.2 => constants.%return.patt.45a
// CHECK:STDOUT:   %.loc4_23.5 => constants.%.095
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.efc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:

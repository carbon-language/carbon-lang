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
// CHECK:STDERR: fail_form_param_not_form.carbon:[[@LINE+8]]:6: error: `ref` is only allowed on a runtime binding [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR: fn G(ref x:? i32);
// CHECK:STDERR:      ^~~
// CHECK:STDERR:
// CHECK:STDERR: fail_form_param_not_form.carbon:[[@LINE+4]]:11: error: semantics TODO: `handle invalid parse trees in `check`` [SemanticsTodo]
// CHECK:STDERR: fn G(ref x:? i32);
// CHECK:STDERR:           ^~
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

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  F(form(var i32), x);
}

// --- fail_todo_composite_generic_form.carbon
library "[[@TEST_NAME]]";

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR: fail_todo_composite_generic_form.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `(Core.Form, Core.Form)` to `Core.Form` [ConversionFailure]
  // CHECK:STDERR:   F((form(var i32), form(ref i32)), (x, ref x));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_composite_generic_form.carbon:[[@LINE+7]]:3: note: type `(Core.Form, Core.Form)` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   F((form(var i32), form(ref i32)), (x, ref x));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_composite_generic_form.carbon:[[@LINE-10]]:14: note: initializing generic parameter `Fm` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(generic Fm: Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:              ^~~~~~~~~~~~~
  // CHECK:STDERR:
  F((form(var i32), form(ref i32)), (x, ref x));
}

// --- fail_todo_deduce_generic_form.carbon
library "[[@TEST_NAME]]";

fn F[Fm: Core.Form](v:? Fm);

fn G() {
  var v: i32 = 0;
  // CHECK:STDERR: fail_todo_deduce_generic_form.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `Fm` [DeductionIncomplete]
  // CHECK:STDERR:   F(ref v);
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_todo_deduce_generic_form.carbon:[[@LINE-7]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[Fm: Core.Form](v:? Fm);
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(ref v);
}

// --- fail_form_generic_arg_not_form.carbon
library "[[@TEST_NAME]]";

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR: fail_form_generic_arg_not_form.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `type` to `Core.Form` [ConversionFailure]
  // CHECK:STDERR:   F(i32, x);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_form_generic_arg_not_form.carbon:[[@LINE+7]]:3: note: type `type` does not implement interface `Core.ImplicitAs(Core.Form)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   F(i32, x);
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_form_generic_arg_not_form.carbon:[[@LINE-10]]:14: note: initializing generic parameter `Fm` declared here [InitializingGenericParam]
  // CHECK:STDERR: fn F(generic Fm: Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:              ^~~~~~~~~~~~~
  // CHECK:STDERR:
  F(i32, x);
}

// --- fail_form_generic_missing_ref_tag.carbon
library "[[@TEST_NAME]]";

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR: fail_form_generic_missing_ref_tag.carbon:[[@LINE+8]]:20: error: argument to `ref` parameter not marked with `ref` [RefParamNoRefTag]
  // CHECK:STDERR:   F(form(ref i32), x);
  // CHECK:STDERR:                    ^
  // CHECK:STDERR: fail_form_generic_missing_ref_tag.carbon: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fail_form_generic_missing_ref_tag.carbon:[[@LINE-8]]:36: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F(generic Fm: Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:                                    ^~~~~~
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
fn F(generic Form: Core.Form) {
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

// --- val_return_form_definition.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F() ->? form(val i32) {
  return 1;
}
//@dump-sem-ir-end

// --- ref_return_class.carbon
library "[[@TEST_NAME]]";

class C {}

//@dump-sem-ir-begin
fn F() -> ref C;
//@dump-sem-ir-end

fn G() {
  //@dump-sem-ir-begin
  F();
  //@dump-sem-ir-end
}

// --- val_return_class.carbon
library "[[@TEST_NAME]]";

class C {}

//@dump-sem-ir-begin
fn F() ->? form(val C);
//@dump-sem-ir-end

fn G() {
  //@dump-sem-ir-begin
  F();
  //@dump-sem-ir-end
}

// --- ref_return_class_generic.carbon
library "[[@TEST_NAME]]";

class C {}

interface I {
  fn F(self) -> ref C;
}

fn G[T: I](x: T) {
  //@dump-sem-ir-begin
  x.F();
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

fn F(generic Fm: Core.Form, v:? Fm) ->? Fm {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.ae1: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.df9 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.df9 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.ae1] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %i32 [concrete = constants.%.ae1]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.df9 = var_pattern %x.param_patt [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.df9 = wrapper_binding_pattern x, %x.var_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %self.param_patt.59b: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cc8: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt.59b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc4 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.df9 = var_pattern %x.param_patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.df9 = wrapper_binding_pattern x, %x.var_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.4e1] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = init_form %i32 [concrete = constants.%.4e1]
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
// CHECK:STDOUT:   %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc4_7.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_7.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_7.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc4: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage @F.%x.var_patt
// CHECK:STDOUT:   assign %x.var, %.loc4
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%x.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %x.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc4(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc4(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc4(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.f09: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.df9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.df9 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.df9 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.df9 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.f09] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = value_form %i32 [concrete = constants.%.f09]
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
// CHECK:STDOUT:   %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_5.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc9_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc9_5.2: %i32 = converted %int_0, %.loc9_5.1 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%.loc9_5.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_component_needs_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %.9f9: Core.Form = ref_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.ae1: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.f28: %pattern_type.df9 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.df9 = return_slot_pattern %.f28, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.df9 = ref_binding_pattern _ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.df9 = ref_return_pattern [concrete = constants.%.f28]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = ref_form %i32 [concrete = constants.%.ae1]
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
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: ref %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: ref %i32 = wrapper_binding _, %F.call
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.df9 = ref_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- var_return_form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.df9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.df9 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.df9 = ref_binding_pattern _ [concrete]
// CHECK:STDOUT:   %_.var_patt: %pattern_type.df9 = var_pattern %_.patt [concrete]
// CHECK:STDOUT:   %self.param_patt.59b: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cc8: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt.59b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc9 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.df9 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %return.param_patt, %.loc4_17.2 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.1: Core.Form = init_form %i32 [concrete = constants.%.4e1]
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
// CHECK:STDOUT:   %_.var: ref %i32 = var_storage %_.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.ref()
// CHECK:STDOUT:   assign %_.var, %F.call
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: ref %i32 = wrapper_binding _, %_.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.df9 = ref_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:     %_.var_patt: %pattern_type.df9 = var_pattern %_.patt [concrete = constants.%_.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %_.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%_.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_return_form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.f09: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.bf7: %pattern_type.df9 = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.df9 = return_slot_pattern %.bf7, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.df9 = value_binding_pattern _ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.df9 = value_return_pattern [concrete = constants.%.bf7]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = value_form %i32 [concrete = constants.%.f09]
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
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: %i32 = call %F.ref()
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %_: %i32 = wrapper_binding _, %F.call
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %_.patt: %pattern_type.df9 = value_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_return_form_definition.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.f09: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.bf7: %pattern_type.df9 = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.36d: %pattern_type.df9 = return_slot_pattern %.bf7, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.874: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.df9 = value_return_pattern [concrete = constants.%.bf7]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt.36d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = value_form %i32 [concrete = constants.%.f09]
// CHECK:STDOUT:     %.loc4_17.3: type = type_component_of %.loc4_17.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.4: %i32 = value_return
// CHECK:STDOUT:     %return: %i32 = return_slot %.loc4_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> val %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc5_11.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_11.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_11.2(%int_1) [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc5_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.874]
// CHECK:STDOUT:   %.loc5_11.2: %i32 = converted %int_1, %.loc5_11.1 [concrete = constants.%int_1.874]
// CHECK:STDOUT:   return %.loc5_11.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_return_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.723: Core.Form = ref_form %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.af6: %pattern_type = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type = return_slot_pattern %.af6, %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc6_11.1: %pattern_type = ref_return_pattern [concrete = constants.%.af6]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern %.loc6_11.1, %.loc6_11.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc6_11.2: Core.Form = ref_form %C.ref [concrete = constants.%.723]
// CHECK:STDOUT:     %.loc6_11.3: type = type_component_of %.loc6_11.2 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc6_11.4: ref %C = ref_return
// CHECK:STDOUT:     %return: ref %C = return_slot %.loc6_11.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> ref %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: ref %C = call %F.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_return_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.fbb: Core.Form = value_form %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.483: %pattern_type = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type = return_slot_pattern %.483, %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc6_17.1: %pattern_type = value_return_pattern [concrete = constants.%.483]
// CHECK:STDOUT:     %return.patt: %pattern_type = return_slot_pattern %.loc6_17.1, %.loc6_17.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc6_17.2: Core.Form = value_form %C.ref [concrete = constants.%.fbb]
// CHECK:STDOUT:     %.loc6_17.3: type = type_component_of %.loc6_17.2 [concrete = constants.%C]
// CHECK:STDOUT:     %.loc6_17.4: %C = value_return
// CHECK:STDOUT:     %return: %C = return_slot %.loc6_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> val %C;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %F.call: %C = call %F.ref()
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_return_class_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %I.type: type = facet_type <@I> [concrete]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I [concrete]
// CHECK:STDOUT:   %assoc0: %I.assoc_type = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %I.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.6cb = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %I.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.b3a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt: %pattern_type.b3a = wrapper_binding_pattern x, %x.param_patt [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.418: type = fn_type @I.WithSelf.F, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.81d: type = fn_type_with_self_type %I.WithSelf.F.type.418, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.81d = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc9_7.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.loc9_7.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.418)]
// CHECK:STDOUT:   %.loc11_4: type = fn_type_with_self_type %I.WithSelf.F.type, %T.loc9_7.1 [symbolic = %.loc11_4 (constants.%.81d)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_7.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc11_4.2: @G.%.loc11_4 (%.81d) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_4.2: <specific function> = specific_impl_function %impl.elem0.loc11_4.2, @I.WithSelf.F(%T.loc9_7.1) [symbolic = %specific_impl_fn.loc11_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @G.%T.as_type.loc9_15.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @G.%T.as_type.loc9_15.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11_4.1: @G.%.loc11_4 (%.81d) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:     %bound_method.loc11_4: <bound method> = bound_method %x.ref, %impl.elem0.loc11_4.1
// CHECK:STDOUT:     %.loc11_7.1: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc9_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_7.2: %I.type = converted constants.%T.as_type, constants.%T [symbolic = %T.loc9_7.1 (constants.%T)]
// CHECK:STDOUT:     %specific_impl_fn.loc11_4.1: <specific function> = specific_impl_function %impl.elem0.loc11_4.1, @I.WithSelf.F(constants.%T) [symbolic = %specific_impl_fn.loc11_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:     %bound_method.loc11_7: <bound method> = bound_method %x.ref, %specific_impl_fn.loc11_4.1
// CHECK:STDOUT:     %I.WithSelf.F.call: ref %C = call %bound_method.loc11_7(%x.ref)
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc9_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc9_15.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b3a
// CHECK:STDOUT:   %x.param_patt.loc9_13.2 => constants.%x.param_patt
// CHECK:STDOUT:   %x.patt.loc9_13.2 => constants.%x.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- form_generic_return.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.13f: type = pattern_type Core.Form [concrete]
// CHECK:STDOUT:   %Fm.patt: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic]
// CHECK:STDOUT:   %Fm: Core.Form = symbolic_binding Fm, 0 [symbolic]
// CHECK:STDOUT:   %.42f: type = type_component_of %Fm [symbolic]
// CHECK:STDOUT:   %pattern_type.177: type = pattern_type %.42f [symbolic]
// CHECK:STDOUT:   %.481d89.1: %pattern_type.177 = splice_inst @F.%.loc4_30.3 [template]
// CHECK:STDOUT:   %v.patt.db46d6.1: %pattern_type.177 = wrapper_binding_pattern v, %.481d89.1 [template]
// CHECK:STDOUT:   %.05fee9.1: %pattern_type.177 = splice_inst @F.%.loc4_37.3 [template]
// CHECK:STDOUT:   %return.patt.94bfcb.1: %pattern_type.177 = return_slot_pattern %.05fee9.1, %.42f [template]
// CHECK:STDOUT:   %.59eb0c.1: %.42f = splice_inst @F.%.loc4_30.5 [template]
// CHECK:STDOUT:   %.4d6629.1: %.42f = splice_inst @F.%.loc4_37.5 [template]
// CHECK:STDOUT:   %.c5c: <instruction> = form_param_pattern_action %Fm, v [template]
// CHECK:STDOUT:   %.481d89.2: %pattern_type.177 = splice_inst %.c5c [template]
// CHECK:STDOUT:   %v.patt.db46d6.2: %pattern_type.177 = wrapper_binding_pattern v, %.481d89.2 [template]
// CHECK:STDOUT:   %.2e1: <instruction> = out_form_param_pattern_action %Fm [template]
// CHECK:STDOUT:   %.05fee9.2: %pattern_type.177 = splice_inst %.2e1 [template]
// CHECK:STDOUT:   %return.patt.94bfcb.2: %pattern_type.177 = return_slot_pattern %.05fee9.2, %.42f [template]
// CHECK:STDOUT:   %.0c9: <instruction> = callee_pattern_match_action %.481d89.1, call_param0 [template]
// CHECK:STDOUT:   %.59eb0c.2: %.42f = splice_inst %.0c9 [template]
// CHECK:STDOUT:   %.ed0: <instruction> = callee_pattern_match_action %.05fee9.1, call_param1 [template]
// CHECK:STDOUT:   %.4d6629.2: %.42f = splice_inst %.ed0 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.c9b: <witness> = require_complete_type %.42f [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %v.patt.c51: %pattern_type.df9 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.437: %pattern_type.df9 = var_pattern %v.patt.c51 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.973abb.1: %pattern_type.df9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.df9 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.df9 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %v.param_patt.cecf16.1: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %v.var_patt.d59ab4.1: %pattern_type.df9 = var_pattern %v.param_patt.cecf16.1 [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.5d4: %pattern_type.df9 = splice_block %v.var_patt.d59ab4.2 [concrete = %v.var_patt.d59ab4.1] {
// CHECK:STDOUT:       %v.param_patt.cecf16.3: %pattern_type.df9 = ref_param_pattern [concrete = %v.param_patt.cecf16.1]
// CHECK:STDOUT:       %v.var_patt.d59ab4.2: %pattern_type.df9 = var_pattern %v.param_patt.cecf16.3 [concrete = %v.var_patt.d59ab4.1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.abc: %pattern_type.df9 = wrapper_binding_pattern v, %v.var_patt.d59ab4.1 [concrete]
// CHECK:STDOUT:   %inst.out_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %return.param_patt.973abb.2: %pattern_type.df9 = out_param_pattern [concrete = %return.param_patt.973abb.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.a21: %pattern_type.df9 = return_slot_pattern %return.param_patt.973abb.1, %i32 [concrete]
// CHECK:STDOUT:   %.bc3: %i32 = splice_inst %.0c9 [template]
// CHECK:STDOUT:   %.b99: %i32 = splice_inst %.ed0 [template]
// CHECK:STDOUT:   %F.specific_fn.e4e: <specific function> = specific_function %F, @F(%.4e1) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.df9 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %.ae1: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %inst.ref_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %v.param_patt.cecf16.2: %pattern_type.df9 = ref_param_pattern [concrete = %v.param_patt.cecf16.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.e90: %pattern_type.df9 = wrapper_binding_pattern v, %v.param_patt.cecf16.1 [concrete]
// CHECK:STDOUT:   %.f28e50.1: %pattern_type.df9 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %inst.ref_return_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f28e50.2: %pattern_type.df9 = ref_return_pattern [concrete = %.f28e50.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.122: %pattern_type.df9 = return_slot_pattern %.f28e50.1, %i32 [concrete]
// CHECK:STDOUT:   %F.specific_fn.8ab: <specific function> = specific_function %F, @F(%.ae1) [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.df9 = value_binding_pattern z [concrete]
// CHECK:STDOUT:   %.f09: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %v.param_patt.dd22fa.1: %pattern_type.df9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %inst.value_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %v.param_patt.dd22fa.2: %pattern_type.df9 = value_param_pattern [concrete = %v.param_patt.dd22fa.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.9ce: %pattern_type.df9 = wrapper_binding_pattern v, %v.param_patt.dd22fa.1 [concrete]
// CHECK:STDOUT:   %.bf73aa.1: %pattern_type.df9 = value_return_pattern [concrete]
// CHECK:STDOUT:   %inst.value_return_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.bf73aa.2: %pattern_type.df9 = value_return_pattern [concrete = %.bf73aa.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.36d: %pattern_type.df9 = return_slot_pattern %.bf73aa.1, %i32 [concrete]
// CHECK:STDOUT:   %F.specific_fn.11f: <specific function> = specific_function %F, @F(%.f09) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4e5: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.4a0: %Destroy.WithSelf.SelfDestruct.type.4e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.71d [concrete]
// CHECK:STDOUT:   %self.param_patt.59b: %pattern_type.df9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cc8: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt.59b [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f: <witness> = custom_witness (%Destroy.WithSelf.Op.403, %Destroy.WithSelf.SubobjectDestroy.d01, %Destroy.WithSelf.SelfDestruct.db3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d43: %Destroy.type = facet_value %i32, (%custom_witness.f8f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e7c: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.d43) [concrete]
// CHECK:STDOUT:   %.eb3: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e7c, %Destroy.facet.d43 [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.e39: %Destroy.assoc_type = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc2]
// CHECK:STDOUT:   %Core.import_ref.71d: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.4e5) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.4a0)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.2 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.df9 = ref_param_pattern [concrete = constants.%self.param_patt.59b]
// CHECK:STDOUT:     %self.patt: %pattern_type.df9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cc8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %Fm.patt.loc4_16.1: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic = %Fm.patt.loc4_16.2 (constants.%Fm.patt)]
// CHECK:STDOUT:     %.loc4_30.2: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_30.3 [template = %.loc4_30.4 (constants.%.481d89.1)]
// CHECK:STDOUT:     %v.patt.loc4_30.1: @F.%pattern_type (%pattern_type.177) = wrapper_binding_pattern v, %.loc4_30.2 [template = %v.patt.loc4_30.2 (constants.%v.patt.db46d6.1)]
// CHECK:STDOUT:     %.loc4_37.2: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_37.3 [template = %.loc4_37.4 (constants.%.05fee9.1)]
// CHECK:STDOUT:     %return.patt.loc4_37.1: @F.%pattern_type (%pattern_type.177) = return_slot_pattern %.loc4_37.2, %.loc4_41 [template = %return.patt.loc4_37.2 (constants.%return.patt.94bfcb.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Fm.ref.loc4_41: Core.Form = name_ref Fm, %Fm.loc4_16.2 [symbolic = %Fm.loc4_16.1 (constants.%Fm)]
// CHECK:STDOUT:     %.loc4_41: type = type_component_of %Fm.ref.loc4_41 [symbolic = %.loc4_33.1 (constants.%.42f)]
// CHECK:STDOUT:     %.loc4_22: type = splice_block %Form.ref [concrete = Core.Form] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Form.ref: type = name_ref Form, imports.%Core.Form [concrete = Core.Form]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Fm.loc4_16.2: Core.Form = symbolic_binding Fm, 0 [symbolic = %Fm.loc4_16.1 (constants.%Fm)]
// CHECK:STDOUT:     %.loc4_30.1: @F.%.loc4_33.1 (%.42f) = splice_inst %.loc4_30.5 [template = %.loc4_30.6 (constants.%.59eb0c.1)]
// CHECK:STDOUT:     %.loc4_33.2: Core.Form = splice_block %Fm.ref.loc4_33 [symbolic = %Fm.loc4_16.1 (constants.%Fm)] {
// CHECK:STDOUT:       %Fm.ref.loc4_33: Core.Form = name_ref Fm, %Fm.loc4_16.2 [symbolic = %Fm.loc4_16.1 (constants.%Fm)]
// CHECK:STDOUT:       %.loc4_33.3: type = type_component_of %Fm.ref.loc4_33 [symbolic = %.loc4_33.1 (constants.%.42f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @F.%.loc4_33.1 (%.42f) = wrapper_binding v, %.loc4_30.1
// CHECK:STDOUT:     %.loc4_37.1: @F.%.loc4_33.1 (%.42f) = splice_inst %.loc4_37.5 [template = %.loc4_37.6 (constants.%.4d6629.1)]
// CHECK:STDOUT:     %return: @F.%.loc4_33.1 (%.42f) = return_slot %.loc4_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%Fm.loc4_16.2: Core.Form) {
// CHECK:STDOUT:   %Fm.patt.loc4_16.2: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic = %Fm.patt.loc4_16.2 (constants.%Fm.patt)]
// CHECK:STDOUT:   %Fm.loc4_16.1: Core.Form = symbolic_binding Fm, 0 [symbolic = %Fm.loc4_16.1 (constants.%Fm)]
// CHECK:STDOUT:   %.loc4_33.1: type = type_component_of %Fm.loc4_16.1 [symbolic = %.loc4_33.1 (constants.%.42f)]
// CHECK:STDOUT:   %.loc4_30.3: <instruction> = form_param_pattern_action %Fm.ref.loc4_33, v [template]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc4_33.1 [symbolic = %pattern_type (constants.%pattern_type.177)]
// CHECK:STDOUT:   %.loc4_30.4: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_30.3 [template = %.loc4_30.4 (constants.%.481d89.1)]
// CHECK:STDOUT:   %v.patt.loc4_30.2: @F.%pattern_type (%pattern_type.177) = wrapper_binding_pattern v, %.loc4_30.4 [template = %v.patt.loc4_30.2 (constants.%v.patt.db46d6.1)]
// CHECK:STDOUT:   %.loc4_37.3: <instruction> = out_form_param_pattern_action %Fm.ref.loc4_41 [template]
// CHECK:STDOUT:   %.loc4_37.4: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc4_37.3 [template = %.loc4_37.4 (constants.%.05fee9.1)]
// CHECK:STDOUT:   %return.patt.loc4_37.2: @F.%pattern_type (%pattern_type.177) = return_slot_pattern %.loc4_37.4, %.loc4_33.1 [template = %return.patt.loc4_37.2 (constants.%return.patt.94bfcb.1)]
// CHECK:STDOUT:   %.loc4_30.5: <instruction> = callee_pattern_match_action constants.%.481d89.1, call_param0 [template]
// CHECK:STDOUT:   %.loc4_30.6: @F.%.loc4_33.1 (%.42f) = splice_inst %.loc4_30.5 [template = %.loc4_30.6 (constants.%.59eb0c.1)]
// CHECK:STDOUT:   %.loc4_37.5: <instruction> = callee_pattern_match_action constants.%.05fee9.1, call_param1 [template]
// CHECK:STDOUT:   %.loc4_37.6: @F.%.loc4_33.1 (%.42f) = splice_inst %.loc4_37.5 [template = %.loc4_37.6 (constants.%.4d6629.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc4_33.1 [symbolic = %require_complete (constants.%require_complete.c9b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%.loc4_30.1: @F.%.loc4_33.1 (%.42f)) -> out %.loc4_37.1:? %Fm {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.ref: @F.%.loc4_33.1 (%.42f) = name_ref v, %v
// CHECK:STDOUT:     return %v.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var.loc11: ref %i32 = var_storage %v.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc11: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc11_3.1: <bound method> = bound_method %int_0, %impl.elem0.loc11 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc11: <specific function> = specific_function %impl.elem0.loc11, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_3.2: <bound method> = bound_method %int_0, %specific_fn.loc11 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc11_3.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc11: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   assign %v.var.loc11, %.loc11
// CHECK:STDOUT:   %i32.loc11: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %v: ref %i32 = wrapper_binding v, %v.var.loc11
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.df9 = ref_binding_pattern v [concrete = constants.%v.patt.c51]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.df9 = var_pattern %v.patt [concrete = constants.%v.var_patt.437]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage %x.var_patt
// CHECK:STDOUT:   %F.ref.loc13: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc13_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc13_30: Core.Form = init_form %i32.loc13_34 [concrete = constants.%.4e1]
// CHECK:STDOUT:   %v.ref.loc13: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %F.specific_fn.loc13: <specific function> = specific_function %F.ref.loc13, @F(constants.%.4e1) [concrete = constants.%F.specific_fn.e4e]
// CHECK:STDOUT:   %.loc13_40: %i32 = acquire_value %v.ref.loc13
// CHECK:STDOUT:   %impl.elem0.loc13: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:   %bound_method.loc13_40.1: <bound method> = bound_method %.loc13_40, %impl.elem0.loc13
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_40.2: <bound method> = bound_method %.loc13_40, %specific_fn.loc13
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc13_40.2(%.loc13_40)
// CHECK:STDOUT:   %v.var.1: ref %i32 = var_storage constants.%v.var_patt.d59ab4.1
// CHECK:STDOUT:   assign %v.var.1, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %F.call.loc13: init %i32 = call %F.specific_fn.loc13(%v.var.1)
// CHECK:STDOUT:   assign %x.var, %F.call.loc13
// CHECK:STDOUT:   %i32.loc13_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.df9 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.df9 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc14: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc14_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_34: Core.Form = ref_form %i32.loc14_38 [concrete = constants.%.ae1]
// CHECK:STDOUT:   %v.ref.loc14: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc14_44: %i32 = ref_tag %v.ref.loc14
// CHECK:STDOUT:   %F.specific_fn.loc14: <specific function> = specific_function %F.ref.loc14, @F(constants.%.ae1) [concrete = constants.%F.specific_fn.8ab]
// CHECK:STDOUT:   %F.call.loc14: ref %i32 = call %F.specific_fn.loc14(%v.ref.loc14)
// CHECK:STDOUT:   %i32.loc14_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %y: ref %i32 = wrapper_binding y, %F.call.loc14
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.df9 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc15: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc15_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc15_30: Core.Form = value_form %i32.loc15_34 [concrete = constants.%.f09]
// CHECK:STDOUT:   %v.ref.loc15: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %F.specific_fn.loc15: <specific function> = specific_function %F.ref.loc15, @F(constants.%.f09) [concrete = constants.%F.specific_fn.11f]
// CHECK:STDOUT:   %.loc15_40: %i32 = acquire_value %v.ref.loc15
// CHECK:STDOUT:   %F.call.loc15: %i32 = call %F.specific_fn.loc15(%.loc15_40)
// CHECK:STDOUT:   %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %z: %i32 = wrapper_binding z, %F.call.loc15
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.df9 = value_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %SelfDestruct.ref: %Destroy.assoc_type = name_ref SelfDestruct, imports.%Core.import_ref.e39 [concrete = constants.%assoc2]
// CHECK:STDOUT:   %impl.elem2: %.eb3 = impl_witness_access constants.%custom_witness.f8f, element2 [concrete = constants.%Destroy.WithSelf.SelfDestruct.db3]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %v.var.1, %impl.elem2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.1: init %empty_tuple.type = call %bound_method.1(%v.var.1)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc13: <bound method> = bound_method %x.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc13: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc13(%x.var)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc11: <bound method> = bound_method %v.var.loc11, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc11: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc11(%v.var.loc11)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Fm) {
// CHECK:STDOUT:   %Fm.patt.loc4_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_16.1 => constants.%Fm
// CHECK:STDOUT:   %.loc4_33.1 => constants.%.42f
// CHECK:STDOUT:   %.loc4_30.3 => constants.%.c5c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.177
// CHECK:STDOUT:   %.loc4_30.4 => constants.%.481d89.2
// CHECK:STDOUT:   %v.patt.loc4_30.2 => constants.%v.patt.db46d6.2
// CHECK:STDOUT:   %.loc4_37.3 => constants.%.2e1
// CHECK:STDOUT:   %.loc4_37.4 => constants.%.05fee9.2
// CHECK:STDOUT:   %return.patt.loc4_37.2 => constants.%return.patt.94bfcb.2
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc4_30.6 => constants.%.59eb0c.2
// CHECK:STDOUT:   %.loc4_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc4_37.6 => constants.%.4d6629.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.4e1) {
// CHECK:STDOUT:   %Fm.patt.loc4_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_16.1 => constants.%.4e1
// CHECK:STDOUT:   %.loc4_33.1 => constants.%i32
// CHECK:STDOUT:   %.loc4_30.3 => constants.%inst.splice_block
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df9
// CHECK:STDOUT:   %.loc4_30.4 => constants.%v.var_patt.d59ab4.1
// CHECK:STDOUT:   %v.patt.loc4_30.2 => constants.%v.patt.abc
// CHECK:STDOUT:   %.loc4_37.3 => constants.%inst.out_param_pattern
// CHECK:STDOUT:   %.loc4_37.4 => constants.%return.param_patt.973abb.1
// CHECK:STDOUT:   %return.patt.loc4_37.2 => constants.%return.patt.a21
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc4_30.6 => constants.%.bc3
// CHECK:STDOUT:   %.loc4_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc4_37.6 => constants.%.b99
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.ae1) {
// CHECK:STDOUT:   %Fm.patt.loc4_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_16.1 => constants.%.ae1
// CHECK:STDOUT:   %.loc4_33.1 => constants.%i32
// CHECK:STDOUT:   %.loc4_30.3 => constants.%inst.ref_param_pattern
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df9
// CHECK:STDOUT:   %.loc4_30.4 => constants.%v.param_patt.cecf16.1
// CHECK:STDOUT:   %v.patt.loc4_30.2 => constants.%v.patt.e90
// CHECK:STDOUT:   %.loc4_37.3 => constants.%inst.ref_return_pattern
// CHECK:STDOUT:   %.loc4_37.4 => constants.%.f28e50.1
// CHECK:STDOUT:   %return.patt.loc4_37.2 => constants.%return.patt.122
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc4_30.6 => constants.%.bc3
// CHECK:STDOUT:   %.loc4_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc4_37.6 => constants.%.b99
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.f09) {
// CHECK:STDOUT:   %Fm.patt.loc4_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc4_16.1 => constants.%.f09
// CHECK:STDOUT:   %.loc4_33.1 => constants.%i32
// CHECK:STDOUT:   %.loc4_30.3 => constants.%inst.value_param_pattern
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.df9
// CHECK:STDOUT:   %.loc4_30.4 => constants.%v.param_patt.dd22fa.1
// CHECK:STDOUT:   %v.patt.loc4_30.2 => constants.%v.patt.9ce
// CHECK:STDOUT:   %.loc4_37.3 => constants.%inst.value_return_pattern
// CHECK:STDOUT:   %.loc4_37.4 => constants.%.bf73aa.1
// CHECK:STDOUT:   %return.patt.loc4_37.2 => constants.%return.patt.36d
// CHECK:STDOUT:   %.loc4_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc4_30.6 => constants.%.bc3
// CHECK:STDOUT:   %.loc4_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc4_37.6 => constants.%.b99
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:

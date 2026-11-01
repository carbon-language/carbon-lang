// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/form.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/call/form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/call/form.carbon

// --- core.carbon

package Core library "[[@TEST_NAME]]";

export import library "prelude/parts/form";

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

import Core library "core";

fn F() -> Core.Form;
// CHECK:STDERR:        ╭─┤ fail_param_form_not_constant.carbon:[[@LINE+7]]:10
// CHECK:STDERR: ->     │ fn G(x:? F());
// CHECK:STDERR:        ·          ━┳━
// CHECK:STDERR:        ·           ┗━━┫ error: cannot evaluate form expression
// CHECK:STDERR:        ·              ┃        [FormExprEvaluationFailure]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn G(x:? F());

// --- fail_form_param_not_form.carbon
library "[[@TEST_NAME]]";

// CHECK:STDERR:        ╭─┤ fail_form_param_not_form.carbon:[[@LINE+11]]:10
// CHECK:STDERR: ->     │ fn F(x:? i32);
// CHECK:STDERR:        ·          ┳━━
// CHECK:STDERR:        ·          ┗━━┫ error: cannot implicitly convert expression of type `type` to
// CHECK:STDERR:        ·          │  ┃        `Core.Form`
// CHECK:STDERR:        ·          │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·          ╰────┤ type `type` does not implement interface
// CHECK:STDERR:        ·               │ `Core.ImplicitAs(Core.Form)`
// CHECK:STDERR:        ·               │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F(x:? i32);
// CHECK:STDERR:        ╭─┤ fail_form_param_not_form.carbon:[[@LINE+14]]:6
// CHECK:STDERR: ->     │ fn G(ref x:? i32);
// CHECK:STDERR:        ·      ━┳━
// CHECK:STDERR:        ·       ┗━━┫ error: `ref` is only allowed on a runtime binding
// CHECK:STDERR:        ·          ┃        [ExpectedRuntimeBindingPatternAfterRef]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_form_param_not_form.carbon:[[@LINE+7]]:11
// CHECK:STDERR: ->     │ fn G(ref x:? i32);
// CHECK:STDERR:        ·           ━┳
// CHECK:STDERR:        ·            ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·               ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn G(ref x:? i32);

// --- fail_missing_ref_tag.carbon
library "[[@TEST_NAME]]";

fn F(x:? form(ref i32));

fn G() {
  var y: i32 = 0;
  // CHECK:STDERR:        ╭─┤ fail_missing_ref_tag.carbon:[[@LINE+11]]:5
  // CHECK:STDERR:        │ fn F(x:? form(ref i32));
  // CHECK:STDERR:        ·      ────────┬────────
  // CHECK:STDERR:        ·              ╰──┤ initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(y);
  // CHECK:STDERR:        ·     ┳
  // CHECK:STDERR:        ·     ┗━━┫ error: argument to `ref` parameter not marked with `ref`
  // CHECK:STDERR:        ·        ┃        [RefParamNoRefTag]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(y);
}

// --- form_generic_param.carbon
library "[[@TEST_NAME]]";

import Core library "core";

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  F(form(var i32), x);
}

// --- fail_todo_composite_generic_form.carbon
library "[[@TEST_NAME]]";

import Core library "core";

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR:        ╭─┤ fail_todo_composite_generic_form.carbon:[[@LINE+16]]:3
  // CHECK:STDERR:        │ fn F(generic Fm: Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:        ·              ──────┬──────
  // CHECK:STDERR:        ·                    ╰──┤ initializing generic parameter `Fm` declared here
  // CHECK:STDERR:        ·                       │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F((form(var i32), form(ref i32)), (x, ref x));
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert expression of type
  // CHECK:STDERR:        ·   │  ┃        `(Core.Form, Core.Form)` to `Core.Form`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·   ╰────┤ type `(Core.Form, Core.Form)` does not implement interface
  // CHECK:STDERR:        ·        │ `Core.ImplicitAs(Core.Form)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F((form(var i32), form(ref i32)), (x, ref x));
}

// --- fail_todo_deduce_generic_form.carbon
library "[[@TEST_NAME]]";

import Core library "core";

fn F[Fm: Core.Form](v:? Fm);

fn G() {
  var v: i32 = 0;
  // CHECK:STDERR:        ╭─┤ fail_todo_deduce_generic_form.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn F[Fm: Core.Form](v:? Fm);
  // CHECK:STDERR:        · ──────────────┬─────────────
  // CHECK:STDERR:        ·               ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·                  │ [DeductionGenericHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(ref v);
  // CHECK:STDERR:        ·   ━━━━┳━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: cannot deduce value for generic parameter `Fm`
  // CHECK:STDERR:        ·          ┃        [DeductionIncomplete]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(ref v);
}

// --- fail_form_generic_arg_not_form.carbon
library "[[@TEST_NAME]]";

import Core library "core";

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR:        ╭─┤ fail_form_generic_arg_not_form.carbon:[[@LINE+14]]:3
  // CHECK:STDERR:        │ fn F(generic Fm: Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:        ·              ──────┬──────
  // CHECK:STDERR:        ·                    ╰──┤ initializing generic parameter `Fm` declared here
  // CHECK:STDERR:        ·                       │ [InitializingGenericParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(i32, x);
  // CHECK:STDERR:        ·   ┳━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot implicitly convert expression of type `type` to `Core.Form`
  // CHECK:STDERR:        ·   │  ┃        [ConversionFailure]
  // CHECK:STDERR:        ·   ╰────┤ type `type` does not implement interface `Core.ImplicitAs(Core.Form)`
  // CHECK:STDERR:        ·        │ [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(i32, x);
}

// --- fail_form_generic_missing_ref_tag.carbon
library "[[@TEST_NAME]]";

import Core library "core";

fn F(generic Fm: Core.Form, unused x:? Fm) {}

fn G() {
  var x: i32;
  // CHECK:STDERR:        ╭─┤ fail_form_generic_missing_ref_tag.carbon:[[@LINE+15]]:20
  // CHECK:STDERR:        │ fn F(generic Fm: Core.Form, unused x:? Fm) {}
  // CHECK:STDERR:        ·                                    ───┬──
  // CHECK:STDERR:        ·                                       ╰──┤ initializing function parameter
  // CHECK:STDERR:        ·                                          │ [InCallToFunctionParam]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(form(ref i32), x);
  // CHECK:STDERR:        ·                    ┳
  // CHECK:STDERR:        ·                    ┗━━┫ error: argument to `ref` parameter not marked with `ref`
  // CHECK:STDERR:        ·                       ┃        [RefParamNoRefTag]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR:        ├─┤ fail_form_generic_missing_ref_tag.carbon
  // CHECK:STDERR:        · ╰──┤ initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(form(ref i32), x);
}

// --- fail_todo_local_init_form_binding.carbon
library "[[@TEST_NAME]]";

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_todo_local_init_form_binding.carbon:[[@LINE+7]]:7
  // CHECK:STDERR: ->     │   let x:? form(var i32) = 0;
  // CHECK:STDERR:        ·       ━━━━━━━━┳━━━━━━━━
  // CHECK:STDERR:        ·               ┗━━┫ error: semantics TODO: `support local form bindings`
  // CHECK:STDERR:        ·                  ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let x:? form(var i32) = 0;
}

// --- fail_todo_local_symbolic_form_binding.carbon
library "[[@TEST_NAME]]";

import Core library "core";

// TODO: this code should fail for a different reason, but right now we don't
// have a way to write correct code that reaches the TODO we're exercising here.
fn F(generic Form: Core.Form) {
  // CHECK:STDERR:        ╭─┤ fail_todo_local_symbolic_form_binding.carbon:[[@LINE+7]]:7
  // CHECK:STDERR: ->     │   let y:? Form = 0;
  // CHECK:STDERR:        ·       ━━━━┳━━━
  // CHECK:STDERR:        ·           ┗━━┫ error: semantics TODO: `support local form bindings`
  // CHECK:STDERR:        ·              ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
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

// CHECK:STDERR:        ╭─┤ fail_return_form_not_form.carbon:[[@LINE+11]]:12
// CHECK:STDERR: ->     │ fn F() ->? i32;
// CHECK:STDERR:        ·            ┳━━
// CHECK:STDERR:        ·            ┗━━┫ error: cannot implicitly convert expression of type `type` to
// CHECK:STDERR:        ·            │  ┃        `Core.Form`
// CHECK:STDERR:        ·            │  ┃        [ConversionFailure]
// CHECK:STDERR:        ·            ╰────┤ type `type` does not implement interface
// CHECK:STDERR:        ·                 │ `Core.ImplicitAs(Core.Form)`
// CHECK:STDERR:        ·                 │ [MissingImplInMemberAccessInContext]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() ->? i32;
// CHECK:STDERR:        ╭─┤ fail_return_form_not_form.carbon:[[@LINE+7]]:12
// CHECK:STDERR: ->     │ fn F() ->? ref i32;
// CHECK:STDERR:        ·            ━━━┳━━━
// CHECK:STDERR:        ·               ┗━━┫ error: `ref` tag is not an argument to a `ref` parameter
// CHECK:STDERR:        ·                  ┃        [RefTagNoRefParam]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F() ->? ref i32;

// --- form_generic_return.carbon
library "[[@TEST_NAME]]";

import Core library "core";
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
// CHECK:STDOUT:   %.30e: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.30e] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = ref_form %i32 [concrete = constants.%.30e]
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
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d94 = var_pattern %x.param_patt [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.var_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc4 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = ref_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d94 = var_pattern %x.param_patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.var_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.0b3] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = init_form %i32 [concrete = constants.%.0b3]
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
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc4_7.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc4_7.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc4_7.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc4_7.1: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage @F.%x.var_patt
// CHECK:STDOUT:   assign %x.var, %.loc4_7.1
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.ref(%x.var)
// CHECK:STDOUT:   %Destroy.facet.loc4_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc4_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc4_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc4_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc4_7.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc4_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%x.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc4(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_form_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.acd: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.d94 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc4_15.1: Core.Form = splice_block %.loc4_15.2 [concrete = constants.%.acd] {
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %.loc4_15.2: Core.Form = value_form %i32 [concrete = constants.%.acd]
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
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_5.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_5.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_5.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc9_5.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc9_5.2: %i32 = converted %int_0, %.loc9_5.1 [concrete = constants.%int_0.753]
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
// CHECK:STDOUT:   %.30e: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.b71: %pattern_type.d94 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.d94 = return_slot_pattern %.b71, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.d94 = ref_binding_pattern _ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.d94 = ref_return_pattern [concrete = constants.%.b71]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = ref_form %i32 [concrete = constants.%.30e]
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
// CHECK:STDOUT:     %_.patt: %pattern_type.d94 = ref_binding_pattern _ [concrete = constants.%_.patt]
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
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.d94 = ref_binding_pattern _ [concrete]
// CHECK:STDOUT:   %_.var_patt: %pattern_type.d94 = var_pattern %_.patt [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %.loc4_17.2 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.1: Core.Form = init_form %i32 [concrete = constants.%.0b3]
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
// CHECK:STDOUT:     %_.patt: %pattern_type.d94 = ref_binding_pattern _ [concrete = constants.%_.patt]
// CHECK:STDOUT:     %_.var_patt: %pattern_type.d94 = var_pattern %_.patt [concrete = constants.%_.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.loc9_7.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_7.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc9_7.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_7.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_7.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc9_7.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%_.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_return_form.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.acd: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.45b: %pattern_type.d94 = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.d94 = return_slot_pattern %.45b, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %_.patt: %pattern_type.d94 = value_binding_pattern _ [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.d94 = value_return_pattern [concrete = constants.%.45b]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = value_form %i32 [concrete = constants.%.acd]
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
// CHECK:STDOUT:     %_.patt: %pattern_type.d94 = value_binding_pattern _ [concrete = constants.%_.patt]
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
// CHECK:STDOUT:   %.acd: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.45b: %pattern_type.d94 = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.f8b: %pattern_type.d94 = return_slot_pattern %.45b, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %.loc4_17.1: %pattern_type.d94 = value_return_pattern [concrete = constants.%.45b]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %.loc4_17.1, %.loc4_17.3 [concrete = constants.%return.patt.f8b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.2: Core.Form = value_form %i32 [concrete = constants.%.acd]
// CHECK:STDOUT:     %.loc4_17.3: type = type_component_of %.loc4_17.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_17.4: %i32 = value_return
// CHECK:STDOUT:     %return: %i32 = return_slot %.loc4_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> val %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc5_11.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_11.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc5_11.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_11.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc5_11.2: %i32 = converted %int_1, %.loc5_11.1 [concrete = constants.%int_1.68a]
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
// CHECK:STDOUT:   %I.WithSelf.F.type.eae: type = fn_type @I.WithSelf.F, @I.WithSelf(%T) [symbolic]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T, @I [symbolic]
// CHECK:STDOUT:   %.b94: type = fn_type_with_self_type %I.WithSelf.F.type.eae, %T [symbolic]
// CHECK:STDOUT:   %impl.elem0: %.b94 = impl_witness_access %I.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn: <specific function> = specific_impl_function %impl.elem0, @I.WithSelf.F(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc9_7.2: %I.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %I.WithSelf.F.type: type = fn_type @I.WithSelf.F, @I.WithSelf(%T.loc9_7.1) [symbolic = %I.WithSelf.F.type (constants.%I.WithSelf.F.type.eae)]
// CHECK:STDOUT:   %.loc11_4: type = fn_type_with_self_type %I.WithSelf.F.type, %T.loc9_7.1 [symbolic = %.loc11_4 (constants.%.b94)]
// CHECK:STDOUT:   %I.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_7.1, @I [symbolic = %I.lookup_impl_witness (constants.%I.lookup_impl_witness)]
// CHECK:STDOUT:   %impl.elem0.loc11_4.2: @G.%.loc11_4 (%.b94) = impl_witness_access %I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_4.2 (constants.%impl.elem0)]
// CHECK:STDOUT:   %specific_impl_fn.loc11_4.2: <specific function> = specific_impl_function %impl.elem0.loc11_4.2, @I.WithSelf.F(%T.loc9_7.1) [symbolic = %specific_impl_fn.loc11_4.2 (constants.%specific_impl_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @G.%T.as_type.loc9_15.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @G.%T.as_type.loc9_15.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %F.ref: %I.assoc_type = name_ref F, @I.WithSelf.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:     %impl.elem0.loc11_4.1: @G.%.loc11_4 (%.b94) = impl_witness_access constants.%I.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc11_4.2 (constants.%impl.elem0)]
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
// CHECK:STDOUT:   %.481d89.1: %pattern_type.177 = splice_inst @F.%.loc6_30.3 [template]
// CHECK:STDOUT:   %v.patt.db46d6.1: %pattern_type.177 = wrapper_binding_pattern v, %.481d89.1 [template]
// CHECK:STDOUT:   %.05fee9.1: %pattern_type.177 = splice_inst @F.%.loc6_37.3 [template]
// CHECK:STDOUT:   %return.patt.94bfcb.1: %pattern_type.177 = return_slot_pattern %.05fee9.1, %.42f [template]
// CHECK:STDOUT:   %.59eb0c.1: %.42f = splice_inst @F.%.loc6_30.5 [template]
// CHECK:STDOUT:   %.4d6629.1: %.42f = splice_inst @F.%.loc6_37.5 [template]
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
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %v.patt.1f6: %pattern_type.d94 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.afe: %pattern_type.d94 = var_pattern %v.patt.1f6 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.2999fd.1: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete]
// CHECK:STDOUT:   %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete]
// CHECK:STDOUT:   %v.param_patt.8c11bb.1: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %v.var_patt.90aa0b.1: %pattern_type.d94 = var_pattern %v.param_patt.8c11bb.1 [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b51: %pattern_type.d94 = splice_block %v.var_patt.90aa0b.2 [concrete = %v.var_patt.90aa0b.1] {
// CHECK:STDOUT:       %v.param_patt.8c11bb.3: %pattern_type.d94 = ref_param_pattern [concrete = %v.param_patt.8c11bb.1]
// CHECK:STDOUT:       %v.var_patt.90aa0b.2: %pattern_type.d94 = var_pattern %v.param_patt.8c11bb.3 [concrete = %v.var_patt.90aa0b.1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.4c9: %pattern_type.d94 = wrapper_binding_pattern v, %v.var_patt.90aa0b.1 [concrete]
// CHECK:STDOUT:   %inst.out_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %return.param_patt.2999fd.2: %pattern_type.d94 = out_param_pattern [concrete = %return.param_patt.2999fd.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.2999fd.1, %i32 [concrete]
// CHECK:STDOUT:   %.c443: %i32 = splice_inst %.0c9 [template]
// CHECK:STDOUT:   %.26e: %i32 = splice_inst %.ed0 [template]
// CHECK:STDOUT:   %F.specific_fn.aed: <specific function> = specific_function %F, @F(%.0b3) [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %y.patt: %pattern_type.d94 = ref_binding_pattern y [concrete]
// CHECK:STDOUT:   %.30e: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %inst.ref_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %v.param_patt.8c11bb.2: %pattern_type.d94 = ref_param_pattern [concrete = %v.param_patt.8c11bb.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.600: %pattern_type.d94 = wrapper_binding_pattern v, %v.param_patt.8c11bb.1 [concrete]
// CHECK:STDOUT:   %.b71e7d.1: %pattern_type.d94 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %inst.ref_return_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b71e7d.2: %pattern_type.d94 = ref_return_pattern [concrete = %.b71e7d.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.2c5: %pattern_type.d94 = return_slot_pattern %.b71e7d.1, %i32 [concrete]
// CHECK:STDOUT:   %F.specific_fn.503: <specific function> = specific_function %F, @F(%.30e) [concrete]
// CHECK:STDOUT:   %z.patt: %pattern_type.d94 = value_binding_pattern z [concrete]
// CHECK:STDOUT:   %.acd: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %v.param_patt.9d4532.1: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %inst.value_param_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %v.param_patt.9d4532.2: %pattern_type.d94 = value_param_pattern [concrete = %v.param_patt.9d4532.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.patt.58a: %pattern_type.d94 = wrapper_binding_pattern v, %v.param_patt.9d4532.1 [concrete]
// CHECK:STDOUT:   %.45b17a.1: %pattern_type.d94 = value_return_pattern [concrete]
// CHECK:STDOUT:   %inst.value_return_pattern: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.45b17a.2: %pattern_type.d94 = value_return_pattern [concrete = %.45b17a.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %return.patt.f8b: %pattern_type.d94 = return_slot_pattern %.45b17a.1, %i32 [concrete]
// CHECK:STDOUT:   %F.specific_fn.efd: <specific function> = specific_function %F, @F(%.acd) [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.687: type = fn_type @SubobjectDestroy.WithSelf.Op.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f: %SubobjectDestroy.WithSelf.Op.type.687 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.49a: %SubobjectDestroy.type = facet_value %i32, (%custom_witness.b80) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.077: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.49a) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %i32, (%DestroyImpl.impl_witness.077, %custom_witness.b80) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Form = %Core.Form
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//core
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Form: type = import_ref Core//prelude/parts/form, Form, loaded [concrete = Core.Form]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %Fm.patt.loc6_16.1: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic = %Fm.patt.loc6_16.2 (constants.%Fm.patt)]
// CHECK:STDOUT:     %.loc6_30.2: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc6_30.3 [template = %.loc6_30.4 (constants.%.481d89.1)]
// CHECK:STDOUT:     %v.patt.loc6_30.1: @F.%pattern_type (%pattern_type.177) = wrapper_binding_pattern v, %.loc6_30.2 [template = %v.patt.loc6_30.2 (constants.%v.patt.db46d6.1)]
// CHECK:STDOUT:     %.loc6_37.2: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc6_37.3 [template = %.loc6_37.4 (constants.%.05fee9.1)]
// CHECK:STDOUT:     %return.patt.loc6_37.1: @F.%pattern_type (%pattern_type.177) = return_slot_pattern %.loc6_37.2, %.loc6_41 [template = %return.patt.loc6_37.2 (constants.%return.patt.94bfcb.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Fm.ref.loc6_41: Core.Form = name_ref Fm, %Fm.loc6_16.2 [symbolic = %Fm.loc6_16.1 (constants.%Fm)]
// CHECK:STDOUT:     %.loc6_41: type = type_component_of %Fm.ref.loc6_41 [symbolic = %.loc6_33.1 (constants.%.42f)]
// CHECK:STDOUT:     %.loc6_22: type = splice_block %Form.ref [concrete = Core.Form] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Form.ref: type = name_ref Form, imports.%Core.Form [concrete = Core.Form]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Fm.loc6_16.2: Core.Form = symbolic_binding Fm, 0 [symbolic = %Fm.loc6_16.1 (constants.%Fm)]
// CHECK:STDOUT:     %.loc6_30.1: @F.%.loc6_33.1 (%.42f) = splice_inst %.loc6_30.5 [template = %.loc6_30.6 (constants.%.59eb0c.1)]
// CHECK:STDOUT:     %.loc6_33.2: Core.Form = splice_block %Fm.ref.loc6_33 [symbolic = %Fm.loc6_16.1 (constants.%Fm)] {
// CHECK:STDOUT:       %Fm.ref.loc6_33: Core.Form = name_ref Fm, %Fm.loc6_16.2 [symbolic = %Fm.loc6_16.1 (constants.%Fm)]
// CHECK:STDOUT:       %.loc6_33.3: type = type_component_of %Fm.ref.loc6_33 [symbolic = %.loc6_33.1 (constants.%.42f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: @F.%.loc6_33.1 (%.42f) = wrapper_binding v, %.loc6_30.1
// CHECK:STDOUT:     %.loc6_37.1: @F.%.loc6_33.1 (%.42f) = splice_inst %.loc6_37.5 [template = %.loc6_37.6 (constants.%.4d6629.1)]
// CHECK:STDOUT:     %return: @F.%.loc6_33.1 (%.42f) = return_slot %.loc6_37.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%Fm.loc6_16.2: Core.Form) {
// CHECK:STDOUT:   %Fm.patt.loc6_16.2: %pattern_type.13f = symbolic_binding_pattern Fm, 0 [symbolic = %Fm.patt.loc6_16.2 (constants.%Fm.patt)]
// CHECK:STDOUT:   %Fm.loc6_16.1: Core.Form = symbolic_binding Fm, 0 [symbolic = %Fm.loc6_16.1 (constants.%Fm)]
// CHECK:STDOUT:   %.loc6_33.1: type = type_component_of %Fm.loc6_16.1 [symbolic = %.loc6_33.1 (constants.%.42f)]
// CHECK:STDOUT:   %.loc6_30.3: <instruction> = form_param_pattern_action %Fm.ref.loc6_33, v [template]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc6_33.1 [symbolic = %pattern_type (constants.%pattern_type.177)]
// CHECK:STDOUT:   %.loc6_30.4: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc6_30.3 [template = %.loc6_30.4 (constants.%.481d89.1)]
// CHECK:STDOUT:   %v.patt.loc6_30.2: @F.%pattern_type (%pattern_type.177) = wrapper_binding_pattern v, %.loc6_30.4 [template = %v.patt.loc6_30.2 (constants.%v.patt.db46d6.1)]
// CHECK:STDOUT:   %.loc6_37.3: <instruction> = out_form_param_pattern_action %Fm.ref.loc6_41 [template]
// CHECK:STDOUT:   %.loc6_37.4: @F.%pattern_type (%pattern_type.177) = splice_inst %.loc6_37.3 [template = %.loc6_37.4 (constants.%.05fee9.1)]
// CHECK:STDOUT:   %return.patt.loc6_37.2: @F.%pattern_type (%pattern_type.177) = return_slot_pattern %.loc6_37.4, %.loc6_33.1 [template = %return.patt.loc6_37.2 (constants.%return.patt.94bfcb.1)]
// CHECK:STDOUT:   %.loc6_30.5: <instruction> = callee_pattern_match_action constants.%.481d89.1, call_param0 [template]
// CHECK:STDOUT:   %.loc6_30.6: @F.%.loc6_33.1 (%.42f) = splice_inst %.loc6_30.5 [template = %.loc6_30.6 (constants.%.59eb0c.1)]
// CHECK:STDOUT:   %.loc6_37.5: <instruction> = callee_pattern_match_action constants.%.05fee9.1, call_param1 [template]
// CHECK:STDOUT:   %.loc6_37.6: @F.%.loc6_33.1 (%.42f) = splice_inst %.loc6_37.5 [template = %.loc6_37.6 (constants.%.4d6629.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc6_33.1 [symbolic = %require_complete (constants.%require_complete.c9b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%.loc6_30.1: @F.%.loc6_33.1 (%.42f)) -> out %.loc6_37.1:? %Fm {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.ref: @F.%.loc6_33.1 (%.42f) = name_ref v, %v
// CHECK:STDOUT:     return %v.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %v.var.loc13: ref %i32 = var_storage %v.var_patt
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc13: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc13_3.1: <bound method> = bound_method %int_0, %impl.elem0.loc13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc13: <specific function> = specific_function %impl.elem0.loc13, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_3.2: <bound method> = bound_method %int_0, %specific_fn.loc13 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_3.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc13_3.1: init %i32 = converted %int_0, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   assign %v.var.loc13, %.loc13_3.1
// CHECK:STDOUT:   %i32.loc13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %v: ref %i32 = wrapper_binding v, %v.var.loc13
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.d94 = ref_binding_pattern v [concrete = constants.%v.patt.1f6]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.d94 = var_pattern %v.patt [concrete = constants.%v.var_patt.afe]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %x.var: ref %i32 = var_storage %x.var_patt
// CHECK:STDOUT:   %F.ref.loc15: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc15_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc15_30: Core.Form = init_form %i32.loc15_34 [concrete = constants.%.0b3]
// CHECK:STDOUT:   %v.ref.loc15: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %F.specific_fn.loc15: <specific function> = specific_function %F.ref.loc15, @F(constants.%.0b3) [concrete = constants.%F.specific_fn.aed]
// CHECK:STDOUT:   %.loc15_40: %i32 = acquire_value %v.ref.loc15
// CHECK:STDOUT:   %impl.elem0.loc15: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:   %bound_method.loc15_40.1: <bound method> = bound_method %.loc15_40, %impl.elem0.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_40.2: <bound method> = bound_method %.loc15_40, %specific_fn.loc15
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc15_40.2(%.loc15_40)
// CHECK:STDOUT:   %v.var.1: ref %i32 = var_storage constants.%v.var_patt.90aa0b.1
// CHECK:STDOUT:   assign %v.var.1, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %F.call.loc15: init %i32 = call %F.specific_fn.loc15(%v.var.1)
// CHECK:STDOUT:   assign %x.var, %F.call.loc15
// CHECK:STDOUT:   %i32.loc15_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %x: ref %i32 = wrapper_binding x, %x.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %x.patt: %pattern_type.d94 = ref_binding_pattern x [concrete = constants.%x.patt]
// CHECK:STDOUT:     %x.var_patt: %pattern_type.d94 = var_pattern %x.patt [concrete = constants.%x.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc16: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc16_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc16_34: Core.Form = ref_form %i32.loc16_38 [concrete = constants.%.30e]
// CHECK:STDOUT:   %v.ref.loc16: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %.loc16_44: %i32 = ref_tag %v.ref.loc16
// CHECK:STDOUT:   %F.specific_fn.loc16: <specific function> = specific_function %F.ref.loc16, @F(constants.%.30e) [concrete = constants.%F.specific_fn.503]
// CHECK:STDOUT:   %F.call.loc16: ref %i32 = call %F.specific_fn.loc16(%v.ref.loc16)
// CHECK:STDOUT:   %i32.loc16_21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %y: ref %i32 = wrapper_binding y, %F.call.loc16
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %y.patt: %pattern_type.d94 = ref_binding_pattern y [concrete = constants.%y.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.ref.loc17: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %i32.loc17_34: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc17_30: Core.Form = value_form %i32.loc17_34 [concrete = constants.%.acd]
// CHECK:STDOUT:   %v.ref.loc17: ref %i32 = name_ref v, %v
// CHECK:STDOUT:   %F.specific_fn.loc17: <specific function> = specific_function %F.ref.loc17, @F(constants.%.acd) [concrete = constants.%F.specific_fn.efd]
// CHECK:STDOUT:   %.loc17_40: %i32 = acquire_value %v.ref.loc17
// CHECK:STDOUT:   %F.call.loc17: %i32 = call %F.specific_fn.loc17(%.loc17_40)
// CHECK:STDOUT:   %i32.loc17_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %z: %i32 = wrapper_binding z, %F.call.loc17
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %z.patt: %pattern_type.d94 = value_binding_pattern z [concrete = constants.%z.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.facet.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.1: %Destroy.type = converted constants.%i32, %Destroy.facet.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.2: %Destroy.type = converted constants.%i32, %Destroy.facet.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.1: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.1: init %empty_tuple.type = call %SelfDestruct.specific_fn.1(%v.var.1)
// CHECK:STDOUT:   %Destroy.facet.loc15_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc15_3.1: %Destroy.type = converted constants.%i32, %Destroy.facet.loc15_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc15_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc15_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc15_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.2: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.loc15: init %empty_tuple.type = call %SelfDestruct.specific_fn.2(%x.var)
// CHECK:STDOUT:   %Destroy.facet.loc13_3.1: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.2: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_3.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc13_3.2: %Destroy.type = facet_value constants.%i32, (constants.%DestroyImpl.impl_witness.077, constants.%custom_witness.b80) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc13_3.3: %Destroy.type = converted constants.%i32, %Destroy.facet.loc13_3.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn.3: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call.loc13: init %empty_tuple.type = call %SelfDestruct.specific_fn.3(%v.var.loc13)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Fm) {
// CHECK:STDOUT:   %Fm.patt.loc6_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc6_16.1 => constants.%Fm
// CHECK:STDOUT:   %.loc6_33.1 => constants.%.42f
// CHECK:STDOUT:   %.loc6_30.3 => constants.%.c5c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.177
// CHECK:STDOUT:   %.loc6_30.4 => constants.%.481d89.2
// CHECK:STDOUT:   %v.patt.loc6_30.2 => constants.%v.patt.db46d6.2
// CHECK:STDOUT:   %.loc6_37.3 => constants.%.2e1
// CHECK:STDOUT:   %.loc6_37.4 => constants.%.05fee9.2
// CHECK:STDOUT:   %return.patt.loc6_37.2 => constants.%return.patt.94bfcb.2
// CHECK:STDOUT:   %.loc6_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc6_30.6 => constants.%.59eb0c.2
// CHECK:STDOUT:   %.loc6_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc6_37.6 => constants.%.4d6629.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.0b3) {
// CHECK:STDOUT:   %Fm.patt.loc6_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc6_16.1 => constants.%.0b3
// CHECK:STDOUT:   %.loc6_33.1 => constants.%i32
// CHECK:STDOUT:   %.loc6_30.3 => constants.%inst.splice_block
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %.loc6_30.4 => constants.%v.var_patt.90aa0b.1
// CHECK:STDOUT:   %v.patt.loc6_30.2 => constants.%v.patt.4c9
// CHECK:STDOUT:   %.loc6_37.3 => constants.%inst.out_param_pattern
// CHECK:STDOUT:   %.loc6_37.4 => constants.%return.param_patt.2999fd.1
// CHECK:STDOUT:   %return.patt.loc6_37.2 => constants.%return.patt.6fa
// CHECK:STDOUT:   %.loc6_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc6_30.6 => constants.%.c443
// CHECK:STDOUT:   %.loc6_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc6_37.6 => constants.%.26e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.30e) {
// CHECK:STDOUT:   %Fm.patt.loc6_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc6_16.1 => constants.%.30e
// CHECK:STDOUT:   %.loc6_33.1 => constants.%i32
// CHECK:STDOUT:   %.loc6_30.3 => constants.%inst.ref_param_pattern
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %.loc6_30.4 => constants.%v.param_patt.8c11bb.1
// CHECK:STDOUT:   %v.patt.loc6_30.2 => constants.%v.patt.600
// CHECK:STDOUT:   %.loc6_37.3 => constants.%inst.ref_return_pattern
// CHECK:STDOUT:   %.loc6_37.4 => constants.%.b71e7d.1
// CHECK:STDOUT:   %return.patt.loc6_37.2 => constants.%return.patt.2c5
// CHECK:STDOUT:   %.loc6_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc6_30.6 => constants.%.c443
// CHECK:STDOUT:   %.loc6_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc6_37.6 => constants.%.26e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.acd) {
// CHECK:STDOUT:   %Fm.patt.loc6_16.2 => constants.%Fm.patt
// CHECK:STDOUT:   %Fm.loc6_16.1 => constants.%.acd
// CHECK:STDOUT:   %.loc6_33.1 => constants.%i32
// CHECK:STDOUT:   %.loc6_30.3 => constants.%inst.value_param_pattern
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %.loc6_30.4 => constants.%v.param_patt.9d4532.1
// CHECK:STDOUT:   %v.patt.loc6_30.2 => constants.%v.patt.58a
// CHECK:STDOUT:   %.loc6_37.3 => constants.%inst.value_return_pattern
// CHECK:STDOUT:   %.loc6_37.4 => constants.%.45b17a.1
// CHECK:STDOUT:   %return.patt.loc6_37.2 => constants.%return.patt.f8b
// CHECK:STDOUT:   %.loc6_30.5 => constants.%.0c9
// CHECK:STDOUT:   %.loc6_30.6 => constants.%.c443
// CHECK:STDOUT:   %.loc6_37.5 => constants.%.ed0
// CHECK:STDOUT:   %.loc6_37.6 => constants.%.26e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:

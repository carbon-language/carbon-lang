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

fn F() {
  // TODO: initialize a_ptr
  //@dump-sem-ir-begin
  var a_ptr: __fn_ptr(form(val i32)) ->? form(var i32);
  var unused result: i32 = a_ptr(0);
  //@dump-sem-ir-end
}

// --- fail_fn_ptr_call_mismatch.carbon

library "[[@TEST_NAME]]";

fn F() {
  // TODO: initialize a_ptr
  var a_ptr: __fn_ptr(form(val i32)) ->? form(var i32);
  // CHECK:STDERR: fail_fn_ptr_call_mismatch.carbon:[[@LINE+7]]:34: error: cannot implicitly convert expression of type `Core.FloatLiteral` to `i32` [ConversionFailure]
  // CHECK:STDERR:   var unused result: i32 = a_ptr(1.0);
  // CHECK:STDERR:                                  ^~~
  // CHECK:STDERR: fail_fn_ptr_call_mismatch.carbon:[[@LINE+4]]:34: note: type `Core.FloatLiteral` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   var unused result: i32 = a_ptr(1.0);
  // CHECK:STDERR:                                  ^~~
  // CHECK:STDERR:
  var unused result: i32 = a_ptr(1.0);
}

// CHECK:STDOUT: --- fn_ptr_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.c2a: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.Form) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%.c2a) [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %.600: type = fn_ptr_type (%.c2a), %.795f [concrete]
// CHECK:STDOUT:   %pattern_type.3c5: type = pattern_type %.600 [concrete]
// CHECK:STDOUT:   %a_ptr.patt: %pattern_type.3c5 = ref_binding_pattern a_ptr [concrete]
// CHECK:STDOUT:   %a_ptr.var_patt: %pattern_type.3c5 = var_pattern %a_ptr.patt [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.type.150: type = fn_type @T.as.DefaultOrUnformed.impl.Op, @T.as.DefaultOrUnformed.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.53a: %T.as.DefaultOrUnformed.impl.Op.type.150 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.9bd: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.856, @T.as.DefaultOrUnformed.impl(%.600) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %.600, (%DefaultOrUnformed.impl_witness.9bd) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
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
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc7_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc6 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cc8: @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op.type (%T.as.DefaultOrUnformed.impl.Op.type.150) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as.DefaultOrUnformed.impl.%T.as.DefaultOrUnformed.impl.Op (constants.%T.as.DefaultOrUnformed.impl.Op.53a)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.856 = impl_witness_table (%Core.import_ref.cc8), @T.as.DefaultOrUnformed.impl [concrete]
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
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value constants.%.600, (constants.%DefaultOrUnformed.impl_witness.9bd) [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %.loc6_55.1: %DefaultOrUnformed.type = converted constants.%.600, %DefaultOrUnformed.facet [concrete = constants.%DefaultOrUnformed.facet]
// CHECK:STDOUT:   %as_type: type = facet_access_type %.loc6_55.1 [concrete = constants.%.600]
// CHECK:STDOUT:   %.loc6_55.2: type = converted %.loc6_55.1, %as_type [concrete = constants.%.600]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %T.as.DefaultOrUnformed.impl.Op.call: init %.600 = call %T.as.DefaultOrUnformed.impl.Op.specific_fn()
// CHECK:STDOUT:   assign %a_ptr.var, %T.as.DefaultOrUnformed.impl.Op.call
// CHECK:STDOUT:   %.loc6_14.1: type = splice_block %.loc6_14.2 [concrete = constants.%.600] {
// CHECK:STDOUT:     %i32.loc6_32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_28: Core.Form = value_form %i32.loc6_32 [concrete = constants.%.c2a]
// CHECK:STDOUT:     %.loc6_36: %tuple.type = tuple_literal (%.loc6_28) [concrete = constants.%tuple]
// CHECK:STDOUT:     %i32.loc6_51: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_47: Core.Form = init_form %i32.loc6_51 [concrete = constants.%.795f]
// CHECK:STDOUT:     %.loc6_14.2: type = fn_ptr_type (%.loc6_28), %.loc6_47 [concrete = constants.%.600]
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
// CHECK:STDOUT:   %.loc7_28: %.600 = acquire_value %a_ptr.ref
// CHECK:STDOUT:   %impl.elem0: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc7_34.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_34.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc7_34.2(%int_0) [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc7_34.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %.loc7_34.2: %i32 = converted %int_0, %.loc7_34.1 [concrete = constants.%int_0.3c0]
// CHECK:STDOUT:   %__fn_ptr_thunk.call: init %i32 = call generated.%__fn_ptr_thunk.decl(%.loc7_28, %.loc7_34.2)
// CHECK:STDOUT:   assign %result.var, %__fn_ptr_thunk.call
// CHECK:STDOUT:   %i32.loc7: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %result: ref %i32 = wrapper_binding result, %result.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %result.patt: %pattern_type.6b6 = ref_binding_pattern result [concrete = constants.%result.patt]
// CHECK:STDOUT:     %result.var_patt: %pattern_type.6b6 = var_pattern %result.patt [concrete = constants.%result.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.bound.loc7: <bound method> = bound_method %result.var, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call.loc7: init %empty_tuple.type = call %Destroy.Op.bound.loc7(%result.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc6: <bound method> = bound_method %a_ptr.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call.loc6: init %empty_tuple.type = call %Destroy.Op.bound.loc6(%a_ptr.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc6(%self.param: ref %.600) = "no_op";
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/array.carbon

// --- type_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: array(T, 3)) -> T { return a[0]; }

fn G() -> C {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- bound_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- type_and_bound.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type, N:! Core.IntLiteral()](a: array(T, N)) {}

fn G() {
  var a: array(C, 3) = ({}, {}, {});
  F(a);
}

// --- fail_bound_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: array(T, 2)) -> T { return a[0]; }

fn G() -> C {
  // TODO: We succeed at deducing T here but fail to convert. Is this the right behavior?
  var a: array(C, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(C, 3)` to `array(C, 2)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+7]]:12: note: type `array(C, 3)` does not implement interface `Core.ImplicitAs(array(C, 2))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE-11]]:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[T:! type](a: array(T, 2)) -> T { return a[0]; }
  // CHECK:STDERR:                ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  // TODO: We succeed at deducing N here but fail to convert. Is this the right behavior?
  var a: array(D, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(D, 3)` to `array(C, 3)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:12: note: type `array(D, 3)` does not implement interface `Core.ImplicitAs(array(C, 3))` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE-11]]:29: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[N:! Core.IntLiteral()](a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR:                             ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_bound_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! i32](a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  // TODO: This fails because the array bound in `F` is effectively
  //   `N.(ImplicitAs(IntLiteral).Convert)()`
  // which we can't deduce through. We should decide if we want to support
  // deductions of that form. If not, it'd be nice to diagnose this situation
  // better.
  // CHECK:STDERR: fail_bound_type_mismatch.carbon:[[@LINE+7]]:10: error: cannot deduce value for generic parameter `N` [DeductionIncomplete]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_bound_type_mismatch.carbon:[[@LINE-12]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[N:! i32](a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// CHECK:STDOUT: --- type_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.743: type = array_type %int_3, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.06f: <witness> = require_complete_type %array_type.743 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_29.2 (%array_type.743) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_29.2 (%array_type.743) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_35: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_29.2 (%array_type.743) = value_param call_param0
// CHECK:STDOUT:     %.loc6_29: type = splice_block %array_type.loc6_29.1 [symbolic = %array_type.loc6_29.2 (constants.%array_type.743)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type.loc6_29.1: type = array_type %int_3, %T.ref.loc6_25 [symbolic = %array_type.loc6_29.2 (constants.%array_type.743)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_29.2 (%array_type.743) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %array_type.loc6_29.2: type = array_type constants.%int_3, %T.loc6_6.2 [symbolic = %array_type.loc6_29.2 (constants.%array_type.743)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_32 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%array_type.loc6_29.2 (%array_type.743) [symbolic = %require_complete.loc6_17 (constants.%require_complete.06f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type](%a.param_patt: @F.%array_type.loc6_29.2 (%array_type.743)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_29.2 (%array_type.743) = name_ref a, %a
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc6_48.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_48.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc6_48.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc6_48.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc6_48.2: %i32 = converted %int_0, %.loc6_48.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc6_49.1: ref @F.%array_type.loc6_29.2 (%array_type.743) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_49.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_49.1, %.loc6_48.2
// CHECK:STDOUT:     %.loc6_49.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_49.2
// CHECK:STDOUT:     return %.loc6_49.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc9_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C = class_init (), %.loc9_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C = class_init (), %.loc9_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C = class_init (), %.loc9_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.002 = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3.2: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.2
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc9 [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc8: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc10: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%.loc10) to %.loc8
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %array_type.loc6_29.2 => constants.%array_type.743
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %array_type.loc6_29.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.dd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.6a2: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d82: <witness> = require_complete_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.41f: <bound method> = bound_method %N, %Convert.956 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.ad4: <bound method> = bound_method %N, %Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.ad4(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc6_6.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_42.2 (%array_type.6a2) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_42.2 (%array_type.6a2) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_26.1: type = splice_block %.loc6_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.3: type = converted %int_literal.make_type, %.loc6_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_42.2 (%array_type.6a2) = value_param call_param0
// CHECK:STDOUT:     %.loc6_42: type = splice_block %array_type.loc6_42.1 [symbolic = %array_type.loc6_42.2 (constants.%array_type.6a2)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_41: Core.IntLiteral = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_42.1: type = array_type %N.ref.loc6_41, %C.ref [symbolic = %array_type.loc6_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_42.2 (%array_type.6a2) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_6.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc6_42.2: type = array_type %N.loc6_6.2, constants.%C [symbolic = %array_type.loc6_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc6_42.2 (%array_type.6a2) [symbolic = %require_complete (constants.%require_complete.d82)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc6_6.2, constants.%Convert.956 [symbolic = %Convert.bound (constants.%Convert.bound.41f)]
// CHECK:STDOUT:   %bound_method.loc6_62.3: <bound method> = bound_method %N.loc6_6.2, constants.%Convert.specific_fn [symbolic = %bound_method.loc6_62.3 (constants.%bound_method.ad4)]
// CHECK:STDOUT:   %int.convert_checked.loc6_62.2: init %i32 = call %bound_method.loc6_62.3(%N.loc6_6.2) [symbolic = %int.convert_checked.loc6_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.loc6_6.1: Core.IntLiteral](%a.param_patt: @F.%array_type.loc6_42.2 (%array_type.6a2)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_61: Core.IntLiteral = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc6_62.1: <bound method> = bound_method %N.ref.loc6_61, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound.41f)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_62.2: <bound method> = bound_method %N.ref.loc6_61, %specific_fn [symbolic = %bound_method.loc6_62.3 (constants.%bound_method.ad4)]
// CHECK:STDOUT:     %int.convert_checked.loc6_62.1: init %i32 = call %bound_method.loc6_62.2(%N.ref.loc6_61) [symbolic = %int.convert_checked.loc6_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc6_62.1: %i32 = value_of_initializer %int.convert_checked.loc6_62.1 [symbolic = %int.convert_checked.loc6_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc6_62.2: %i32 = converted %N.ref.loc6_61, %.loc6_62.1 [symbolic = %int.convert_checked.loc6_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     return %.loc6_62.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc9_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C = class_init (), %.loc9_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C = class_init (), %.loc9_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C = class_init (), %.loc9_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.002 = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3.2: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.2
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10_12: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc10_12)
// CHECK:STDOUT:   %.loc10_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc10_14.2: %i32 = converted %F.call, %.loc10_14.1
// CHECK:STDOUT:   return %.loc10_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%N.patt
// CHECK:STDOUT:   %array_type.loc6_42.2 => constants.%array_type.6a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%int_3.1ba
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%N.patt
// CHECK:STDOUT:   %array_type.loc6_42.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.b30
// CHECK:STDOUT:   %bound_method.loc6_62.3 => constants.%bound_method.047
// CHECK:STDOUT:   %int.convert_checked.loc6_62.2 => constants.%int_3.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %array_type.bb5: type = array_type %N, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.bb5 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C, %int_3) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %N.patt.loc6_16.1: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_52.2 (%array_type.bb5) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_52.2 (%array_type.bb5) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %.loc6_36.1: type = splice_block %.loc6_36.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.3: type = converted %int_literal.make_type, %.loc6_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_16.1: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_52.2 (%array_type.bb5) = value_param call_param0
// CHECK:STDOUT:     %.loc6_52: type = splice_block %array_type.loc6_52.1 [symbolic = %array_type.loc6_52.2 (constants.%array_type.bb5)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc6_16.1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_52.1: type = array_type %N.ref, %T.ref [symbolic = %array_type.loc6_52.2 (constants.%array_type.bb5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_52.2 (%array_type.bb5) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type, %N.loc6_16.1: Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %N.loc6_16.2: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic = %N.loc6_16.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc6_16.2: Core.IntLiteral = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc6_16.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc6_52.2: type = array_type %N.loc6_16.2, %T.loc6_6.2 [symbolic = %array_type.loc6_52.2 (constants.%array_type.bb5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc6_52.2 (%array_type.bb5) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type, %N.patt.loc6_16.1: Core.IntLiteral](%a.param_patt: @F.%array_type.loc6_52.2 (%array_type.bb5)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc9_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C = class_init (), %.loc9_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C = class_init (), %.loc9_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C = class_init (), %.loc9_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.002 = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3.2: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.2
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%int_3) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %N.loc6_16.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%N.patt
// CHECK:STDOUT:   %array_type.loc6_52.2 => constants.%array_type.bb5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%int_3) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %N.loc6_16.2 => constants.%int_3
// CHECK:STDOUT:   %N.patt.loc6_16.2 => constants.%N.patt
// CHECK:STDOUT:   %array_type.loc6_52.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type.9d4: type = array_type %int_2, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.d11: <witness> = require_complete_type %array_type.9d4 [symbolic]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %array_type.15a: type = array_type %int_2, %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.8eb: <witness> = complete_type_witness %array_type.15a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_29.2 (%array_type.9d4) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_29.2 (%array_type.9d4) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc6_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc6_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_35: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_29.2 (%array_type.9d4) = value_param call_param0
// CHECK:STDOUT:     %.loc6_29: type = splice_block %array_type.loc6_29.1 [symbolic = %array_type.loc6_29.2 (constants.%array_type.9d4)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.1 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %array_type.loc6_29.1: type = array_type %int_2, %T.ref.loc6_25 [symbolic = %array_type.loc6_29.2 (constants.%array_type.9d4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_29.2 (%array_type.9d4) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_6.1: type) {
// CHECK:STDOUT:   %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc6_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %array_type.loc6_29.2: type = array_type constants.%int_2, %T.loc6_6.2 [symbolic = %array_type.loc6_29.2 (constants.%array_type.9d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32: <witness> = require_complete_type @F.%T.loc6_6.2 (%T) [symbolic = %require_complete.loc6_32 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type @F.%array_type.loc6_29.2 (%array_type.9d4) [symbolic = %require_complete.loc6_17 (constants.%require_complete.d11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc6_6.1: type](%a.param_patt: @F.%array_type.loc6_29.2 (%array_type.9d4)) -> @F.%T.loc6_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_29.2 (%array_type.9d4) = name_ref a, %a
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc6_48.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_48.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:     %int.convert_checked: init %i32 = call %bound_method.loc6_48.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc6_48.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc6_48.2: %i32 = converted %int_0, %.loc6_48.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:     %.loc6_49.1: ref @F.%array_type.loc6_29.2 (%array_type.9d4) = value_as_ref %a.ref
// CHECK:STDOUT:     %.loc6_49.2: ref @F.%T.loc6_6.2 (%T) = array_index %.loc6_49.1, %.loc6_48.2
// CHECK:STDOUT:     %.loc6_49.3: @F.%T.loc6_6.2 (%T) = bind_value %.loc6_49.2
// CHECK:STDOUT:     return %.loc6_49.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param_patt: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc10_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc10_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_35.1: %tuple.type = tuple_literal (%.loc10_26.1, %.loc10_30.1, %.loc10_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %.loc10_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc10_26.2: init %C = class_init (), %.loc10_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.3: init %C = converted %.loc10_26.1, %.loc10_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc10_30.2: init %C = class_init (), %.loc10_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.5: init %C = converted %.loc10_30.1, %.loc10_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc10_34.2: init %C = class_init (), %.loc10_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.7: init %C = converted %.loc10_34.1, %.loc10_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.8: init %array_type.002 = array_init (%.loc10_35.3, %.loc10_35.5, %.loc10_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_3.2: init %array_type.002 = converted %.loc10_35.1, %.loc10_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_3.2
// CHECK:STDOUT:   %.loc10_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc10 [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc8: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc21: %array_type.15a = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(<error>) to %.loc8
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %array_type.loc6_29.2 => constants.%array_type.9d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc6_6.2 => constants.%T.patt
// CHECK:STDOUT:   %array_type.loc6_29.2 => constants.%array_type.15a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.8eb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.6a2: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d82: <witness> = require_complete_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.41f: <bound method> = bound_method %N, %Convert.956 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.ad4: <bound method> = bound_method %N, %Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.ad4(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.fe4: type = array_type %int_3.1ba, %D [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.fe4 = tuple_value (%D.val, %D.val, %D.val) [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/types/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc7_6.1: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_6.2 (constants.%N.patt)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc7_42.2 (%array_type.6a2) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc7_42.2 (%array_type.6a2) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_26.1: type = splice_block %.loc7_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %int_literal.make_type: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.2: type = value_of_initializer %int_literal.make_type [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.3: type = converted %int_literal.make_type, %.loc7_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc7_42.2 (%array_type.6a2) = value_param call_param0
// CHECK:STDOUT:     %.loc7_42: type = splice_block %array_type.loc7_42.1 [symbolic = %array_type.loc7_42.2 (constants.%array_type.6a2)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc7_41: Core.IntLiteral = name_ref N, %N.loc7_6.1 [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc7_42.1: type = array_type %N.ref.loc7_41, %C.ref [symbolic = %array_type.loc7_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc7_42.2 (%array_type.6a2) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc7_6.1: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc7_6.2: Core.IntLiteral = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc7_6.2 (constants.%N.patt)]
// CHECK:STDOUT:   %array_type.loc7_42.2: type = array_type %N.loc7_6.2, constants.%C [symbolic = %array_type.loc7_42.2 (constants.%array_type.6a2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc7_42.2 (%array_type.6a2) [symbolic = %require_complete (constants.%require_complete.d82)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc7_6.2, constants.%Convert.956 [symbolic = %Convert.bound (constants.%Convert.bound.41f)]
// CHECK:STDOUT:   %bound_method.loc7_62.3: <bound method> = bound_method %N.loc7_6.2, constants.%Convert.specific_fn [symbolic = %bound_method.loc7_62.3 (constants.%bound_method.ad4)]
// CHECK:STDOUT:   %int.convert_checked.loc7_62.2: init %i32 = call %bound_method.loc7_62.3(%N.loc7_6.2) [symbolic = %int.convert_checked.loc7_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.loc7_6.1: Core.IntLiteral](%a.param_patt: @F.%array_type.loc7_42.2 (%array_type.6a2)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc7_61: Core.IntLiteral = name_ref N, %N.loc7_6.1 [symbolic = %N.loc7_6.2 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:     %bound_method.loc7_62.1: <bound method> = bound_method %N.ref.loc7_61, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound.41f)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_62.2: <bound method> = bound_method %N.ref.loc7_61, %specific_fn [symbolic = %bound_method.loc7_62.3 (constants.%bound_method.ad4)]
// CHECK:STDOUT:     %int.convert_checked.loc7_62.1: init %i32 = call %bound_method.loc7_62.2(%N.ref.loc7_61) [symbolic = %int.convert_checked.loc7_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc7_62.1: %i32 = value_of_initializer %int.convert_checked.loc7_62.1 [symbolic = %int.convert_checked.loc7_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     %.loc7_62.2: %i32 = converted %N.ref.loc7_61, %.loc7_62.1 [symbolic = %int.convert_checked.loc7_62.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:     return %.loc7_62.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.fe4 = binding_pattern a
// CHECK:STDOUT:     %.loc11_3.1: %array_type.fe4 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.fe4 = var a
// CHECK:STDOUT:   %.loc11_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_35.1: %tuple.type = tuple_literal (%.loc11_26.1, %.loc11_30.1, %.loc11_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_35.2: ref %D = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_26.2: init %D = class_init (), %.loc11_35.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.3: init %D = converted %.loc11_26.1, %.loc11_26.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_35.4: ref %D = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc11_30.2: init %D = class_init (), %.loc11_35.4 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.5: init %D = converted %.loc11_30.1, %.loc11_30.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc11_35.6: ref %D = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc11_34.2: init %D = class_init (), %.loc11_35.6 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.7: init %D = converted %.loc11_34.1, %.loc11_34.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.8: init %array_type.fe4 = array_init (%.loc11_35.3, %.loc11_35.5, %.loc11_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3.2: init %array_type.fe4 = converted %.loc11_35.1, %.loc11_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3.2
// CHECK:STDOUT:   %.loc11_20: type = splice_block %array_type [concrete = constants.%array_type.fe4] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %D.ref [concrete = constants.%array_type.fe4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.fe4 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.fe4 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc22_12: %array_type.002 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %.loc22_14.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc22_14.2: %i32 = converted %F.call, %.loc22_14.1
// CHECK:STDOUT:   return %.loc22_14.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc7_6.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc7_6.2 => constants.%N.patt
// CHECK:STDOUT:   %array_type.loc7_42.2 => constants.%array_type.6a2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc7_6.2 => constants.%int_3.1ba
// CHECK:STDOUT:   %N.patt.loc7_6.2 => constants.%N.patt
// CHECK:STDOUT:   %array_type.loc7_42.2 => constants.%array_type.002
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound.b30
// CHECK:STDOUT:   %bound_method.loc7_62.3 => constants.%bound_method.047
// CHECK:STDOUT:   %int.convert_checked.loc7_62.2 => constants.%int_3.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.8e2: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.2fd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Convert.type.71e: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %impl_witness.023: <witness> = impl_witness (imports.%Core.import_ref.85c), @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.4ad: type = fn_type @Convert.3, @impl.971(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.960: %Convert.type.4ad = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2fd = facet_value %i32, (%impl_witness.023) [concrete]
// CHECK:STDOUT:   %.4d8: type = fn_type_with_self_type %Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.51e, %Convert.960 [symbolic]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.960, @Convert.3(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.51e, %Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %int.convert_checked: init Core.IntLiteral = call %bound_method(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type.b2f: type = array_type %int.convert_checked, %C [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cde: <witness> = require_complete_type %array_type.b2f [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc6_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:     %a.patt: @F.%array_type.loc6_28.2 (%array_type.b2f) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%array_type.loc6_28.2 (%array_type.b2f) = value_param_pattern %a.patt, call_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_28.2 (%array_type.b2f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28: type = splice_block %array_type.loc6_28.1 [symbolic = %array_type.loc6_28.2 (constants.%array_type.b2f)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_27: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0: %.4d8 = impl_witness_access constants.%impl_witness.023, element0 [concrete = constants.%Convert.960]
// CHECK:STDOUT:       %bound_method.loc6_27.1: <bound method> = bound_method %N.ref.loc6_27, %impl.elem0 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.3(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc6_27.2: <bound method> = bound_method %N.ref.loc6_27, %specific_fn [symbolic = %bound_method.loc6_27.3 (constants.%bound_method)]
// CHECK:STDOUT:       %int.convert_checked.loc6_27.1: init Core.IntLiteral = call %bound_method.loc6_27.2(%N.ref.loc6_27) [symbolic = %int.convert_checked.loc6_27.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc6_27.1: Core.IntLiteral = value_of_initializer %int.convert_checked.loc6_27.1 [symbolic = %int.convert_checked.loc6_27.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %.loc6_27.2: Core.IntLiteral = converted %N.ref.loc6_27, %.loc6_27.1 [symbolic = %int.convert_checked.loc6_27.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:       %array_type.loc6_28.1: type = array_type %.loc6_27.2, %C.ref [symbolic = %array_type.loc6_28.2 (constants.%array_type.b2f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_28.2 (%array_type.b2f) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.1: %i32) {
// CHECK:STDOUT:   %N.loc6_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   %N.patt.loc6_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.8e2)]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %N.loc6_6.2, constants.%Convert.960 [symbolic = %Convert.bound (constants.%Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc6_27.3: <bound method> = bound_method %N.loc6_6.2, constants.%Convert.specific_fn [symbolic = %bound_method.loc6_27.3 (constants.%bound_method)]
// CHECK:STDOUT:   %int.convert_checked.loc6_27.2: init Core.IntLiteral = call %bound_method.loc6_27.3(%N.loc6_6.2) [symbolic = %int.convert_checked.loc6_27.2 (constants.%int.convert_checked)]
// CHECK:STDOUT:   %array_type.loc6_28.2: type = array_type %int.convert_checked.loc6_27.2, constants.%C [symbolic = %array_type.loc6_28.2 (constants.%array_type.b2f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%array_type.loc6_28.2 (%array_type.b2f) [symbolic = %require_complete (constants.%require_complete.cde)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.patt.loc6_6.1: %i32](%a.param_patt: @F.%array_type.loc6_28.2 (%array_type.b2f)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_47: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:     return %N.ref.loc6_47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %array_type.002 = binding_pattern a
// CHECK:STDOUT:     %.loc9_3.1: %array_type.002 = var_pattern %a.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var a
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C = class_init (), %.loc9_35.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C = class_init (), %.loc9_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C = class_init (), %.loc9_35.6 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.002 = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) to %a.var [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3.2: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.2
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%N.51e
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%N.patt.8e2
// CHECK:STDOUT:   %Convert.bound => constants.%Convert.bound
// CHECK:STDOUT:   %bound_method.loc6_27.3 => constants.%bound_method
// CHECK:STDOUT:   %int.convert_checked.loc6_27.2 => constants.%int.convert_checked
// CHECK:STDOUT:   %array_type.loc6_28.2 => constants.%array_type.b2f
// CHECK:STDOUT: }
// CHECK:STDOUT:

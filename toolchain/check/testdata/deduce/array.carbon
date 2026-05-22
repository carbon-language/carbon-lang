// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/array.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/array.carbon

// --- type_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: array(T, 3)) -> T { return F(a); }

fn G() -> C {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- bound_only.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! Core.IntLiteral()](unused a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  var a: array(C, 3) = ({}, {}, {});
  return F(a);
}

// --- type_and_bound.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type, N:! Core.IntLiteral()](unused a: array(T, N)) {}

fn G() {
  var a: array(C, 3) = ({}, {}, {});
  F(a);
}

// --- fail_bound_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](a: array(T, 2)) -> T { return F(a); }

fn G() -> C {
  // TODO: We succeed at deducing T here but fail to convert. Is this the right behavior?
  var a: array(C, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(C, 3)` to `array(C, 2)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE+7]]:12: note: type `array(C, 3)` does not implement interface `Core.ImplicitAs(array(C, 2))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_bound_mismatch.carbon:[[@LINE-11]]:16: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[T:! type](a: array(T, 2)) -> T { return F(a); }
  // CHECK:STDERR:                ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}
class D {}

fn F[N:! Core.IntLiteral()](unused a: array(C, N)) -> i32 { return N; }

fn G() -> i32 {
  // TODO: We succeed at deducing N here but fail to convert. Is this the right behavior?
  var a: array(D, 3) = ({}, {}, {});
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+10]]:12: error: cannot implicitly convert expression of type `array(D, 3)` to `array(C, 3)` [ConversionFailure]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+7]]:12: note: type `array(D, 3)` does not implement interface `Core.ImplicitAs(array(C, 3))` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return F(a);
  // CHECK:STDERR:            ^
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE-11]]:36: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR: fn F[N:! Core.IntLiteral()](unused a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR:                                    ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_bound_type_mismatch.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[N:! i32](unused a: array(C, N)) -> i32 { return N; }

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
  // CHECK:STDERR: fn F[N:! i32](unused a: array(C, N)) -> i32 { return N; }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(a);
}

// --- fail_todo_array_length_from_tuple.carbon
library "[[@TEST_NAME]]";

class C {}

fn F[N:! i32](unused a: array(C, N)) {}

fn G() {
  // TODO: Deduce N as 3 from the tuple's size.
  //
  // CHECK:STDERR: fail_todo_array_length_from_tuple.carbon:[[@LINE+7]]:3: error: cannot deduce value for generic parameter `N` [DeductionIncomplete]
  // CHECK:STDERR:   F(({}, {}, {}));
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_array_length_from_tuple.carbon:[[@LINE-8]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[N:! i32](unused a: array(C, N)) {}
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  F(({}, {}, {}));
}

// CHECK:STDOUT: --- type_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.3ec: type = array_type %int_3, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b3f: type = pattern_type %array_type.3ec [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.3ec [symbolic]
// CHECK:STDOUT:   %F.specific_fn.da4: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.3ca: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.d9b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.8a5: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.2fe: type = pattern_type %array_type.8a5 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.8a5 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn.568: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.50c: <witness> = complete_type_witness %array_type.8a5 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc6_17 (%pattern_type.b3f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc6_17 (%pattern_type.b3f) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_35 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_35 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc6_35 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_35: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_35.3: Core.Form = init_form %T.ref.loc6_35 [symbolic = %.loc6_35.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_29.1 (%array_type.3ec) = value_param call_param0
// CHECK:STDOUT:     %.loc6_29: type = splice_block %array_type.loc6_29.2 [symbolic = %array_type.loc6_29.1 (constants.%array_type.3ec)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type.loc6_29.2: type = array_type %int_3, %T.ref.loc6_25 [symbolic = %array_type.loc6_29.1 (constants.%array_type.3ec)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_29.1 (%array_type.3ec) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d9b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.d9b = return_slot_pattern %return.param_patt, %C.ref.loc8 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_11.2: Core.Form = init_form %C.ref.loc8 [concrete = constants.%.3ca]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %array_type.loc6_29.1: type = array_type constants.%int_3, %T.loc6_7.1 [symbolic = %array_type.loc6_29.1 (constants.%array_type.3ec)]
// CHECK:STDOUT:   %pattern_type.loc6_17: type = pattern_type %array_type.loc6_29.1 [symbolic = %pattern_type.loc6_17 (constants.%pattern_type.b3f)]
// CHECK:STDOUT:   %.loc6_35.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_35.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_35: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_35 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_29.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.da4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_29.1 (%array_type.3ec)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_29.1 (%array_type.3ec) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc6_46.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.da4)]
// CHECK:STDOUT:     %.loc6_35.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_35.1 = call %F.specific_fn.loc6_46.1(%a.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.2fe = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.2fe = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.8a5 = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.8a5 to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %array_type.8a5 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.8a5] {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc9 [concrete = constants.%array_type.8a5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.8a5 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.8a5 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.568]
// CHECK:STDOUT:   %.loc8_11.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %.loc10: %array_type.8a5 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %C to %.loc8_11.1 = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %array_type.8a5) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.3ec
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.b3f
// CHECK:STDOUT:   %.loc6_35.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_35 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.da4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.8a5
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.2fe
// CHECK:STDOUT:   %.loc6_35.2 => constants.%.3ca
// CHECK:STDOUT:   %pattern_type.loc6_35 => constants.%pattern_type.d9b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.50c
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.568
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.8a0: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.66d: type = pattern_type %array_type.8a0 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.6f4: <witness> = require_complete_type %array_type.8a0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ad7: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.431: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.431(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.e7a: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %pattern_type.16b: type = pattern_type %array_type.e7a [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.e7a = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.32a: <witness> = complete_type_witness %array_type.e7a [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.353: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.66d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.66d) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_55: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %.loc6_26.1: type = splice_block %.loc6_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.3: type = converted %IntLiteral.call, %.loc6_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_7.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_49.1 (%array_type.8a0) = value_param call_param0
// CHECK:STDOUT:     %.loc6_49: type = splice_block %array_type.loc6_49.2 [symbolic = %array_type.loc6_49.1 (constants.%array_type.8a0)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_48: Core.IntLiteral = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_49.2: type = array_type %N.ref.loc6_48, %C.ref [symbolic = %array_type.loc6_49.1 (constants.%array_type.8a0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_49.1 (%array_type.8a0) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_7.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_7.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc6_49.1: type = array_type %N.loc6_7.1, constants.%C [symbolic = %array_type.loc6_49.1 (constants.%array_type.8a0)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc6_49.1 [symbolic = %pattern_type (constants.%pattern_type.66d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_49.1 [symbolic = %require_complete (constants.%require_complete.6f4)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ad7)]
// CHECK:STDOUT:   %bound_method.loc6_69.3: <bound method> = bound_method %N.loc6_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc6_69.3 (constants.%bound_method.431)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_69.2: init %i32 = call %bound_method.loc6_69.3(%N.loc6_7.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_69.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_49.1 (%array_type.8a0)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_68: Core.IntLiteral = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:     %bound_method.loc6_69.1: <bound method> = bound_method %N.ref.loc6_68, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ad7)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_69.2: <bound method> = bound_method %N.ref.loc6_68, %specific_fn [symbolic = %bound_method.loc6_69.3 (constants.%bound_method.431)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_69.1: init %i32 = call %bound_method.loc6_69.2(%N.ref.loc6_68) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_69.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc6_69: init %i32 = converted %N.ref.loc6_68, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_69.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_69.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc6_69
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.16b = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.16b = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.e7a = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.e7a to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %array_type.e7a = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.e7a] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.e7a]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.e7a = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.e7a = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %array_type.e7a = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %array_type.e7a) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc6_49.1 => constants.%array_type.8a0
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.66d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc6_49.1 => constants.%array_type.e7a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.16b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.32a
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a
// CHECK:STDOUT:   %bound_method.loc6_69.3 => constants.%bound_method.353
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_69.2 => constants.%int_3.49a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 1 [symbolic]
// CHECK:STDOUT:   %array_type.6d2: type = array_type %N, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4d8: type = pattern_type %array_type.6d2 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.6d2 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.357: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.811: type = pattern_type %array_type.357 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.357 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C, %int_3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.c1f: <witness> = complete_type_witness %array_type.357 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.4d8) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.4d8) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc6_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_36.1: type = splice_block %.loc6_36.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.loc6_17: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.3: type = converted %IntLiteral.call, %.loc6_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_17.2: Core.IntLiteral = symbolic_binding N, 1 [symbolic = %N.loc6_17.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_59.1 (%array_type.6d2) = value_param call_param0
// CHECK:STDOUT:     %.loc6_59: type = splice_block %array_type.loc6_59.2 [symbolic = %array_type.loc6_59.1 (constants.%array_type.6d2)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc6_17.2 [symbolic = %N.loc6_17.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_59.2: type = array_type %N.ref, %T.ref [symbolic = %array_type.loc6_59.1 (constants.%array_type.6d2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_59.1 (%array_type.6d2) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type, %N.loc6_17.2: Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc6_17.1: Core.IntLiteral = symbolic_binding N, 1 [symbolic = %N.loc6_17.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc6_59.1: type = array_type %N.loc6_17.1, %T.loc6_7.1 [symbolic = %array_type.loc6_59.1 (constants.%array_type.6d2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc6_59.1 [symbolic = %pattern_type (constants.%pattern_type.4d8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_59.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_59.1 (%array_type.6d2)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.811 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.811 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.357 = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.357 to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %array_type.357 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.357] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.357]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.357 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.357 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%int_3) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %array_type.357 = acquire_value %a.ref
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %array_type.357) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %N.loc6_17.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc6_59.1 => constants.%array_type.6d2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4d8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%int_3) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %N.loc6_17.1 => constants.%int_3
// CHECK:STDOUT:   %array_type.loc6_59.1 => constants.%array_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.811
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.c1f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type.a0b: type = array_type %int_2, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.b42: type = pattern_type %array_type.a0b [symbolic]
// CHECK:STDOUT:   %.184347.1: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.a0b [symbolic]
// CHECK:STDOUT:   %F.specific_fn.0fb: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %.081: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.f30: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.7cf: type = pattern_type %array_type.f30 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type.f30 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %array_type.cae: type = array_type %int_2, %C [concrete]
// CHECK:STDOUT:   %pattern_type.6c2: type = pattern_type %array_type.cae [concrete]
// CHECK:STDOUT:   %F.specific_fn.c32: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc10_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.98d: <witness> = complete_type_witness %array_type.cae [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc6_17 (%pattern_type.b42) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc6_17 (%pattern_type.b42) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_35 (%pattern_type.51d1c4.1) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_35 (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt, %T.ref.loc6_35 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_35: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_35.3: Core.Form = init_form %T.ref.loc6_35 [symbolic = %.loc6_35.2 (constants.%.184347.1)]
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_29.1 (%array_type.a0b) = value_param call_param0
// CHECK:STDOUT:     %.loc6_29: type = splice_block %array_type.loc6_29.2 [symbolic = %array_type.loc6_29.1 (constants.%array_type.a0b)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %array_type.loc6_29.2: type = array_type %int_2, %T.ref.loc6_25 [symbolic = %array_type.loc6_29.1 (constants.%array_type.a0b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_29.1 (%array_type.a0b) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %C.ref.loc8 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_11.2: Core.Form = init_form %C.ref.loc8 [concrete = constants.%.081]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %array_type.loc6_29.1: type = array_type constants.%int_2, %T.loc6_7.1 [symbolic = %array_type.loc6_29.1 (constants.%array_type.a0b)]
// CHECK:STDOUT:   %pattern_type.loc6_17: type = pattern_type %array_type.loc6_29.1 [symbolic = %pattern_type.loc6_17 (constants.%pattern_type.b42)]
// CHECK:STDOUT:   %.loc6_35.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_35.2 (constants.%.184347.1)]
// CHECK:STDOUT:   %pattern_type.loc6_35: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_35 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_29.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.0fb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_29.1 (%array_type.a0b)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_29.1 (%array_type.a0b) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc6_46.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.0fb)]
// CHECK:STDOUT:     %.loc6_35.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_35.1 = call %F.specific_fn.loc6_46.1(%a.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7cf = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.7cf = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.f30 = var %a.var_patt
// CHECK:STDOUT:   %.loc10_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc10_35.1: %tuple.type.8d4 = tuple_literal (%.loc10_26.1, %.loc10_30.1, %.loc10_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc10_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc10_26.2: init %C to %.loc10_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.3: init %C = converted %.loc10_26.1, %.loc10_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc10_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc10_30.2: init %C to %.loc10_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.5: init %C = converted %.loc10_30.1, %.loc10_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc10_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc10_34.2: init %C to %.loc10_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.7: init %C = converted %.loc10_34.1, %.loc10_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc10_35.8: init %array_type.f30 to %a.var = array_init (%.loc10_35.3, %.loc10_35.5, %.loc10_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc10_3: init %array_type.f30 = converted %.loc10_35.1, %.loc10_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_3
// CHECK:STDOUT:   %.loc10_20: type = splice_block %array_type [concrete = constants.%array_type.f30] {
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc10 [concrete = constants.%array_type.f30]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.f30 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.f30 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.c32]
// CHECK:STDOUT:   %.loc8_11.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %.loc21: %array_type.cae = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %C to %.loc8_11.1 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc10_3.3(%self.param: ref %array_type.f30) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.a0b
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.b42
// CHECK:STDOUT:   %.loc6_35.2 => constants.%.184347.1
// CHECK:STDOUT:   %pattern_type.loc6_35 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.0fb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.cae
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.6c2
// CHECK:STDOUT:   %.loc6_35.2 => constants.%.081
// CHECK:STDOUT:   %pattern_type.loc6_35 => constants.%pattern_type.c6b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.98d
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.c32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %array_type.fab: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.b73: type = pattern_type %array_type.fab [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.3a0: <witness> = require_complete_type %array_type.fab [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ad7: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.431: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.431(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.2d1: type = array_type %int_3.1ba, %D [concrete]
// CHECK:STDOUT:   %pattern_type.97c: type = pattern_type %array_type.2d1 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.2d1 = tuple_value (%D.val, %D.val, %D.val) [concrete]
// CHECK:STDOUT:   %array_type.d86: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %pattern_type.4f9: type = pattern_type %array_type.d86 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc11_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.a51: <witness> = complete_type_witness %array_type.d86 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %bound_method.353: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.49a: %i32 = int_value 3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .IntLiteral = %Core.IntLiteral
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.IntLiteral: %IntLiteral.type = import_ref Core//prelude/parts/int_literal, IntLiteral, loaded [concrete = constants.%IntLiteral]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.b73) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.b73) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_55: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %.loc7_26.1: type = splice_block %.loc7_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.3: type = converted %IntLiteral.call, %.loc7_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_7.2: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc7_7.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc7_49.1 (%array_type.fab) = value_param call_param0
// CHECK:STDOUT:     %.loc7_49: type = splice_block %array_type.loc7_49.2 [symbolic = %array_type.loc7_49.1 (constants.%array_type.fab)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc7_48: Core.IntLiteral = name_ref N, %N.loc7_7.2 [symbolic = %N.loc7_7.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc7_49.2: type = array_type %N.ref.loc7_48, %C.ref [symbolic = %array_type.loc7_49.1 (constants.%array_type.fab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc7_49.1 (%array_type.fab) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc7_7.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_7.1: Core.IntLiteral = symbolic_binding N, 0 [symbolic = %N.loc7_7.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc7_49.1: type = array_type %N.loc7_7.1, constants.%C [symbolic = %array_type.loc7_49.1 (constants.%array_type.fab)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc7_49.1 [symbolic = %pattern_type (constants.%pattern_type.b73)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc7_49.1 [symbolic = %require_complete (constants.%require_complete.3a0)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc7_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ad7)]
// CHECK:STDOUT:   %bound_method.loc7_69.3: <bound method> = bound_method %N.loc7_7.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc7_69.3 (constants.%bound_method.431)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_69.2: init %i32 = call %bound_method.loc7_69.3(%N.loc7_7.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_69.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc7_49.1 (%array_type.fab)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc7_68: Core.IntLiteral = name_ref N, %N.loc7_7.2 [symbolic = %N.loc7_7.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:     %bound_method.loc7_69.1: <bound method> = bound_method %N.ref.loc7_68, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.ad7)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_69.2: <bound method> = bound_method %N.ref.loc7_68, %specific_fn [symbolic = %bound_method.loc7_69.3 (constants.%bound_method.431)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_69.1: init %i32 = call %bound_method.loc7_69.2(%N.ref.loc7_68) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_69.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc7_69: init %i32 = converted %N.ref.loc7_68, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_69.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_69.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc7_69
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.97c = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.97c = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.2d1 = var %a.var_patt
// CHECK:STDOUT:   %.loc11_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_35.1: %tuple.type.8d4 = tuple_literal (%.loc11_26.1, %.loc11_30.1, %.loc11_34.1) [concrete = constants.%tuple]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc11_35.2: ref %D = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc11_26.2: init %D to %.loc11_35.2 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.3: init %D = converted %.loc11_26.1, %.loc11_26.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc11_35.4: ref %D = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc11_30.2: init %D to %.loc11_35.4 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.5: init %D = converted %.loc11_30.1, %.loc11_30.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc11_35.6: ref %D = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc11_34.2: init %D to %.loc11_35.6 = class_init () [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.7: init %D = converted %.loc11_34.1, %.loc11_34.2 [concrete = constants.%D.val]
// CHECK:STDOUT:   %.loc11_35.8: init %array_type.2d1 to %a.var = array_init (%.loc11_35.3, %.loc11_35.5, %.loc11_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc11_3: init %array_type.2d1 = converted %.loc11_35.1, %.loc11_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3
// CHECK:STDOUT:   %.loc11_20: type = splice_block %array_type [concrete = constants.%array_type.2d1] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %D.ref [concrete = constants.%array_type.2d1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.2d1 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.2d1 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc22: %array_type.d86 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.3(%self.param: ref %array_type.2d1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc7_7.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc7_49.1 => constants.%array_type.fab
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc7_7.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc7_49.1 => constants.%array_type.d86
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.a51
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.a7a
// CHECK:STDOUT:   %bound_method.loc7_69.3 => constants.%bound_method.353
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_69.2 => constants.%int_3.49a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.afc: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2f8: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.7c2: %Int.as.ImplicitAs.impl.Convert.type.2f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d2b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.07d, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.73a: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.735: %Int.as.ImplicitAs.impl.Convert.type.73a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4bd = facet_value %i32, (%ImplicitAs.impl_witness.d2b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f57: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.516: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f57, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.afc, %Int.as.ImplicitAs.impl.Convert.735 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.735, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.470: <bound method> = bound_method %N.afc, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.470(%N.afc) [symbolic]
// CHECK:STDOUT:   %array_type.82f: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.92e: type = pattern_type %array_type.82f [symbolic]
// CHECK:STDOUT:   %.70c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1f0: <witness> = require_complete_type %array_type.82f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.afc, %Int.as.Copy.impl.Op.c4a [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.8ba: <bound method> = bound_method %N.afc, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.306: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %pattern_type.690: type = pattern_type %array_type.306 [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple.5c1: %tuple.type.8d4 = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.306 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.3: type = fn_type @Destroy.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.3: %Destroy.Op.type.af7ec0.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2f8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.7c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.07d = impl_witness_table (%Core.import_ref.dd0), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.c6b = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.92e) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.92e) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32.loc6_41 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_41: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_41: Core.Form = init_form %i32.loc6_41 [concrete = constants.%.70c]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc6_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.afc)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_35.1 (%array_type.82f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_35: type = splice_block %array_type.loc6_35.2 [symbolic = %array_type.loc6_35.1 (constants.%array_type.82f)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_34: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.afc)]
// CHECK:STDOUT:       %impl.elem0.loc6_34: %.516 = impl_witness_access constants.%ImplicitAs.impl_witness.d2b, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.735]
// CHECK:STDOUT:       %bound_method.loc6_34.2: <bound method> = bound_method %N.ref.loc6_34, %impl.elem0.loc6_34 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn.loc6_34: <specific function> = specific_function %impl.elem0.loc6_34, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc6_34.3: <bound method> = bound_method %N.ref.loc6_34, %specific_fn.loc6_34 [symbolic = %bound_method.loc6_34.1 (constants.%bound_method.470)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc6_34.2: init Core.IntLiteral = call %bound_method.loc6_34.3(%N.ref.loc6_34) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc6_34.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc6_34.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc6_34.2: Core.IntLiteral = converted %N.ref.loc6_34, %.loc6_34.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc6_35.2: type = array_type %.loc6_34.2, %C.ref [symbolic = %array_type.loc6_35.1 (constants.%array_type.82f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_35.1 (%array_type.82f) = value_binding a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c6b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c6b = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32 [concrete = constants.%.70c]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_7.2: %i32) {
// CHECK:STDOUT:   %N.loc6_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.afc)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.ImplicitAs.impl.Convert.735 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc6_34.1: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc6_34.1 (constants.%bound_method.470)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6_34.1: init Core.IntLiteral = call %bound_method.loc6_34.1(%N.loc6_7.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc6_35.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc6_34.1, constants.%C [symbolic = %array_type.loc6_35.1 (constants.%array_type.82f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc6_35.1 [symbolic = %pattern_type (constants.%pattern_type.92e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_35.1 [symbolic = %require_complete (constants.%require_complete.1f0)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.c4a [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc6_54.3: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_54.3 (constants.%bound_method.8ba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_35.1 (%array_type.82f)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_54: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.afc)]
// CHECK:STDOUT:     %impl.elem0.loc6_54: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:     %bound_method.loc6_54.1: <bound method> = bound_method %N.ref.loc6_54, %impl.elem0.loc6_54 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn.loc6_54: <specific function> = specific_function %impl.elem0.loc6_54, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_54.2: <bound method> = bound_method %N.ref.loc6_54, %specific_fn.loc6_54 [symbolic = %bound_method.loc6_54.3 (constants.%bound_method.8ba)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_54.2(%N.ref.loc6_54) [symbolic = %N.loc6_7.1 (constants.%N.afc)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.690 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.690 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.306 = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1) [concrete = constants.%tuple.5c1]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc9_35.2: ref %C = array_index %a.var, %int_0
// CHECK:STDOUT:   %.loc9_26.2: init %C to %.loc9_35.2 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.3: init %C = converted %.loc9_26.1, %.loc9_26.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %.loc9_35.4: ref %C = array_index %a.var, %int_1
// CHECK:STDOUT:   %.loc9_30.2: init %C to %.loc9_35.4 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.5: init %C = converted %.loc9_30.1, %.loc9_30.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:   %.loc9_35.6: ref %C = array_index %a.var, %int_2
// CHECK:STDOUT:   %.loc9_34.2: init %C to %.loc9_35.6 = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.7: init %C = converted %.loc9_34.1, %.loc9_34.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc9_35.8: init %array_type.306 to %a.var = array_init (%.loc9_35.3, %.loc9_35.5, %.loc9_35.7) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc9_3: init %array_type.306 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.306] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.306]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.306 = ref_binding a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.306 = name_ref a, %a
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1dc86d.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_3.3(%self.param: ref %array_type.306) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.afc) {
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%N.afc
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc6_34.1 => constants.%bound_method.470
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6_34.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc6_35.1 => constants.%array_type.82f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.92e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_array_length_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.afc: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.4bd: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = symbolic_binding From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.2f8: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.7c2: %Int.as.ImplicitAs.impl.Convert.type.2f8 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d2b: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.07d, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.73a: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.735: %Int.as.ImplicitAs.impl.Convert.type.73a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.4bd = facet_value %i32, (%ImplicitAs.impl_witness.d2b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f57: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.516: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f57, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.afc, %Int.as.ImplicitAs.impl.Convert.735 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.735, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %N.afc, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method(%N.afc) [symbolic]
// CHECK:STDOUT:   %array_type: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.801: type = pattern_type %array_type [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.f64: <witness> = require_complete_type %array_type [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%empty_struct, %empty_struct, %empty_struct) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd0: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.2f8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.7c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.07d = impl_witness_table (%Core.import_ref.dd0), @Int.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.c6b = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.801) = value_param_pattern [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.801) = at_binding_pattern a, %a.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc5_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc5_7.1 (constants.%N.afc)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc5_35.1 (%array_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_35: type = splice_block %array_type.loc5_35.2 [symbolic = %array_type.loc5_35.1 (constants.%array_type)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref: %i32 = name_ref N, %N.loc5_7.2 [symbolic = %N.loc5_7.1 (constants.%N.afc)]
// CHECK:STDOUT:       %impl.elem0: %.516 = impl_witness_access constants.%ImplicitAs.impl_witness.d2b, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.735]
// CHECK:STDOUT:       %bound_method.loc5_34.2: <bound method> = bound_method %N.ref, %impl.elem0 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc5_34.3: <bound method> = bound_method %N.ref, %specific_fn [symbolic = %bound_method.loc5_34.1 (constants.%bound_method)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc5_34.2: init Core.IntLiteral = call %bound_method.loc5_34.3(%N.ref) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc5_34.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc5_34.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc5_34.2: Core.IntLiteral = converted %N.ref, %.loc5_34.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc5_35.2: type = array_type %.loc5_34.2, %C.ref [symbolic = %array_type.loc5_35.1 (constants.%array_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc5_35.1 (%array_type) = value_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc5_7.2: %i32) {
// CHECK:STDOUT:   %N.loc5_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc5_7.1 (constants.%N.afc)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc5_7.1, constants.%Int.as.ImplicitAs.impl.Convert.735 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc5_34.1: <bound method> = bound_method %N.loc5_7.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc5_34.1 (constants.%bound_method)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_34.1: init Core.IntLiteral = call %bound_method.loc5_34.1(%N.loc5_7.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc5_34.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc5_35.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc5_34.1, constants.%C [symbolic = %array_type.loc5_35.1 (constants.%array_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc5_35.1 [symbolic = %pattern_type (constants.%pattern_type.801)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc5_35.1 [symbolic = %require_complete (constants.%require_complete.f64)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc5_35.1 (%array_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc17_7: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_11: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_15: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc17_16: %tuple.type = tuple_literal (%.loc17_7, %.loc17_11, %.loc17_15) [concrete = constants.%tuple]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.afc) {
// CHECK:STDOUT:   %N.loc5_7.1 => constants.%N.afc
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc5_34.1 => constants.%bound_method
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc5_34.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc5_35.1 => constants.%array_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.801
// CHECK:STDOUT: }
// CHECK:STDOUT:

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

fn F[T:! type](a: array(T, 2)) -> T { return F(a); }

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
  // CHECK:STDERR: fn F[T:! type](a: array(T, 2)) -> T { return F(a); }
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.743: type = array_type %int_3, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.58b: type = pattern_type %array_type.743 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.06f: <witness> = require_complete_type %array_type.743 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %ptr.301: type = ptr_type %array_type.002 [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %array_type.002 [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.002, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.0fd: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.d02: %AggregateT.as_type.as.Destroy.impl.Op.type.0fd = struct_value () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
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
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc6_16 (%pattern_type.58b) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc6_16 (%pattern_type.58b) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_32 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_32 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_35: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_29.1 (%array_type.743) = value_param call_param0
// CHECK:STDOUT:     %.loc6_29: type = splice_block %array_type.loc6_29.2 [symbolic = %array_type.loc6_29.1 (constants.%array_type.743)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:       %array_type.loc6_29.2: type = array_type %int_3, %T.ref.loc6_25 [symbolic = %array_type.loc6_29.1 (constants.%array_type.743)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_29.1 (%array_type.743) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
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
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %array_type.loc6_29.1: type = array_type constants.%int_3, %T.loc6_6.1 [symbolic = %array_type.loc6_29.1 (constants.%array_type.743)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %array_type.loc6_29.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.58b)]
// CHECK:STDOUT:   %pattern_type.loc6_32: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type.loc6_32 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc6_32 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %array_type.loc6_29.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.06f)]
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.1) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_29.1 (%array_type.743)) -> %return.param: @F.%T.loc6_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_29.1 (%array_type.743) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc6_46.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %.loc6_32: ref @F.%T.loc6_6.1 (%T) = splice_block %return {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.1 (%T) = call %F.specific_fn.loc6_46.1(%a.ref) to %.loc6_32
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a63 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a63 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
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
// CHECK:STDOUT:   %.loc9_3.1: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref.loc9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc9 [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %.loc8: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc10: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%.loc10) to %.loc8
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%array_type.002, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_3.2: %type_where = converted constants.%array_type.002, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.d02
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.301 = addr_of %a.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.743
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.58b
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.06f
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%C
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.002
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.a63
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.dd1
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bound_only.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %array_type.6a2: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.9ee: type = pattern_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d82: <witness> = require_complete_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.818: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.775: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.775(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %ptr.301: type = ptr_type %array_type.002 [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %array_type.002 [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.002, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.0fd: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.d02: %AggregateT.as_type.as.Destroy.impl.Op.type.0fd = struct_value () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.9ee) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.9ee) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_26.1: type = splice_block %.loc6_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_26.3: type = converted %IntLiteral.call, %.loc6_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_42.1 (%array_type.6a2) = value_param call_param0
// CHECK:STDOUT:     %.loc6_42: type = splice_block %array_type.loc6_42.2 [symbolic = %array_type.loc6_42.1 (constants.%array_type.6a2)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_41: Core.IntLiteral = name_ref N, %N.loc6_6.2 [symbolic = %N.loc6_6.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_42.2: type = array_type %N.ref.loc6_41, %C.ref [symbolic = %array_type.loc6_42.1 (constants.%array_type.6a2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_42.1 (%array_type.6a2) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT: generic fn @F(%N.loc6_6.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc6_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc6_42.1: type = array_type %N.loc6_6.1, constants.%C [symbolic = %array_type.loc6_42.1 (constants.%array_type.6a2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc6_42.1 [symbolic = %pattern_type (constants.%pattern_type.9ee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_42.1 [symbolic = %require_complete (constants.%require_complete.d82)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_6.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.818)]
// CHECK:STDOUT:   %bound_method.loc6_62.3: <bound method> = bound_method %N.loc6_6.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc6_62.3 (constants.%bound_method.775)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_62.2: init %i32 = call %bound_method.loc6_62.3(%N.loc6_6.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_62.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_42.1 (%array_type.6a2)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_61: Core.IntLiteral = name_ref N, %N.loc6_6.2 [symbolic = %N.loc6_6.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:     %bound_method.loc6_62.1: <bound method> = bound_method %N.ref.loc6_61, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.818)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_62.2: <bound method> = bound_method %N.ref.loc6_61, %specific_fn [symbolic = %bound_method.loc6_62.3 (constants.%bound_method.775)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_62.1: init %i32 = call %bound_method.loc6_62.2(%N.ref.loc6_61) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_62.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc6_62: init %i32 = converted %N.ref.loc6_61, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_62.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_62.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc6_62 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a63 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a63 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
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
// CHECK:STDOUT:   %.loc9_3.1: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc10: %array_type.002 = bind_value %a.ref
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc10)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%array_type.002, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_3.2: %type_where = converted constants.%array_type.002, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.d02
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.301 = addr_of %a.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc6_6.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc6_42.1 => constants.%array_type.6a2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc6_6.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc6_42.1 => constants.%array_type.002
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a63
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595
// CHECK:STDOUT:   %bound_method.loc6_62.3 => constants.%bound_method.f36
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc6_62.2 => constants.%int_3.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- type_and_bound.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %array_type.bb5: type = array_type %N, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.261: type = pattern_type %array_type.bb5 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ec9: <witness> = require_complete_type %array_type.bb5 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %ptr.301: type = ptr_type %array_type.002 [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %array_type.002 [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%C, %int_3) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.002, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.0fd: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.d02: %AggregateT.as_type.as.Destroy.impl.Op.type.0fd = struct_value () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
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
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.261) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.261) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self.1: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_36.1: type = splice_block %.loc6_36.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self.2: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc6_36.3: type = converted %IntLiteral.call, %.loc6_36.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_16.2: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic = %N.loc6_16.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_52.1 (%array_type.bb5) = value_param call_param0
// CHECK:STDOUT:     %.loc6_52: type = splice_block %array_type.loc6_52.2 [symbolic = %array_type.loc6_52.1 (constants.%array_type.bb5)] {
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %N.ref: Core.IntLiteral = name_ref N, %N.loc6_16.2 [symbolic = %N.loc6_16.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc6_52.2: type = array_type %N.ref, %T.ref [symbolic = %array_type.loc6_52.1 (constants.%array_type.bb5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_52.1 (%array_type.bb5) = bind_name a, %a.param
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
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type, %N.loc6_16.2: Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %N.loc6_16.1: Core.IntLiteral = bind_symbolic_name N, 1 [symbolic = %N.loc6_16.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc6_52.1: type = array_type %N.loc6_16.1, %T.loc6_6.1 [symbolic = %array_type.loc6_52.1 (constants.%array_type.bb5)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc6_52.1 [symbolic = %pattern_type (constants.%pattern_type.261)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_52.1 [symbolic = %require_complete (constants.%require_complete.ec9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_52.1 (%array_type.bb5)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a63 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a63 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
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
// CHECK:STDOUT:   %.loc9_3.1: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.1
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
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%array_type.002, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_3.2: %type_where = converted constants.%array_type.002, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.d02
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.301 = addr_of %a.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %N.loc6_16.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc6_52.1 => constants.%array_type.bb5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.261
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%int_3) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%C
// CHECK:STDOUT:   %N.loc6_16.1 => constants.%int_3
// CHECK:STDOUT:   %array_type.loc6_52.1 => constants.%array_type.002
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a63
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array_type.9d4: type = array_type %int_2, %T [symbolic]
// CHECK:STDOUT:   %pattern_type.a4c: type = pattern_type %array_type.9d4 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.d11: <witness> = require_complete_type %array_type.9d4 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %ptr.301: type = ptr_type %array_type.002 [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %array_type.002 [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %array_type.15a: type = array_type %int_2, %C [concrete]
// CHECK:STDOUT:   %pattern_type.114: type = pattern_type %array_type.15a [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.002, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.0fd: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.d02: %AggregateT.as_type.as.Destroy.impl.Op.type.0fd = struct_value () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.8eb: <witness> = complete_type_witness %array_type.15a [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc6_16 (%pattern_type.a4c) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc6_16 (%pattern_type.a4c) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_32 (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_32 (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_35: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc6_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_29.1 (%array_type.9d4) = value_param call_param0
// CHECK:STDOUT:     %.loc6_29: type = splice_block %array_type.loc6_29.2 [symbolic = %array_type.loc6_29.1 (constants.%array_type.9d4)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_6.2 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:       %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2]
// CHECK:STDOUT:       %array_type.loc6_29.2: type = array_type %int_2, %T.ref.loc6_25 [symbolic = %array_type.loc6_29.1 (constants.%array_type.9d4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_29.1 (%array_type.9d4) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8: type = name_ref C, file.%C.decl [concrete = constants.%C]
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
// CHECK:STDOUT: generic fn @F(%T.loc6_6.2: type) {
// CHECK:STDOUT:   %T.loc6_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc6_6.1 (constants.%T)]
// CHECK:STDOUT:   %array_type.loc6_29.1: type = array_type constants.%int_2, %T.loc6_6.1 [symbolic = %array_type.loc6_29.1 (constants.%array_type.9d4)]
// CHECK:STDOUT:   %pattern_type.loc6_16: type = pattern_type %array_type.loc6_29.1 [symbolic = %pattern_type.loc6_16 (constants.%pattern_type.a4c)]
// CHECK:STDOUT:   %pattern_type.loc6_32: type = pattern_type %T.loc6_6.1 [symbolic = %pattern_type.loc6_32 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32: <witness> = require_complete_type %T.loc6_6.1 [symbolic = %require_complete.loc6_32 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %array_type.loc6_29.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.d11)]
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2: <specific function> = specific_function constants.%F, @F(%T.loc6_6.1) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_29.1 (%array_type.9d4)) -> %return.param: @F.%T.loc6_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %a.ref: @F.%array_type.loc6_29.1 (%array_type.9d4) = name_ref a, %a
// CHECK:STDOUT:     %F.specific_fn.loc6_46.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_46.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %.loc6_32: ref @F.%T.loc6_6.1 (%T) = splice_block %return {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_6.1 (%T) = call %F.specific_fn.loc6_46.1(%a.ref) to %.loc6_32
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a63 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a63 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var %a.var_patt
// CHECK:STDOUT:   %.loc10_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc10_35.1: %tuple.type.8d4 = tuple_literal (%.loc10_26.1, %.loc10_30.1, %.loc10_34.1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
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
// CHECK:STDOUT:   %.loc10_3.1: init %array_type.002 = converted %.loc10_35.1, %.loc10_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc10_3.1
// CHECK:STDOUT:   %.loc10_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref.loc10: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref.loc10 [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %.loc8: ref %C = splice_block %return {}
// CHECK:STDOUT:   %.loc21: %array_type.15a = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(<error>) to %.loc8
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%array_type.002, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc10_3.2: %type_where = converted constants.%array_type.002, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.d02
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.301 = addr_of %a.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%T
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.9d4
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.a4c
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.d11
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_6.1 => constants.%C
// CHECK:STDOUT:   %array_type.loc6_29.1 => constants.%array_type.15a
// CHECK:STDOUT:   %pattern_type.loc6_16 => constants.%pattern_type.114
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.c48
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_32 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.8eb
// CHECK:STDOUT:   %F.specific_fn.loc6_46.2 => constants.%F.specific_fn.04a
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
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %array_type.6a2: type = array_type %N, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.9ee: type = pattern_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d82: <witness> = require_complete_type %array_type.6a2 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.204: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.9e9, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.584 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.204) [concrete]
// CHECK:STDOUT:   %.1df: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.818: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.775: <bound method> = bound_method %N, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.775(%N) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.fe4: type = array_type %int_3.1ba, %D [concrete]
// CHECK:STDOUT:   %ptr.af6: type = ptr_type %array_type.fe4 [concrete]
// CHECK:STDOUT:   %pattern_type.f30: type = pattern_type %array_type.fe4 [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %D.val: %D = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.fe4 = tuple_value (%D.val, %D.val, %D.val) [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3.1ba, %C [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %array_type.002 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_3.1ba) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.fe4, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.9be: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.6f0: %AggregateT.as_type.as.Destroy.impl.Op.type.9be = struct_value () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.6f0, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %complete_type.dd1: <witness> = complete_type_witness %array_type.002 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [concrete]
// CHECK:STDOUT:   %bound_method.f36: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.ee7: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.340) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.1c0)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.9e9 = impl_witness_table (%Core.import_ref.ee7), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.9ee) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.9ee) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc7_26.1: type = splice_block %.loc7_26.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, imports.%Core.IntLiteral [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc7_26.3: type = converted %IntLiteral.call, %.loc7_26.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc7_6.2: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_6.1 (constants.%N)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc7_42.1 (%array_type.6a2) = value_param call_param0
// CHECK:STDOUT:     %.loc7_42: type = splice_block %array_type.loc7_42.2 [symbolic = %array_type.loc7_42.1 (constants.%array_type.6a2)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc7_41: Core.IntLiteral = name_ref N, %N.loc7_6.2 [symbolic = %N.loc7_6.1 (constants.%N)]
// CHECK:STDOUT:       %array_type.loc7_42.2: type = array_type %N.ref.loc7_41, %C.ref [symbolic = %array_type.loc7_42.1 (constants.%array_type.6a2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc7_42.1 (%array_type.6a2) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT: generic fn @F(%N.loc7_6.2: Core.IntLiteral) {
// CHECK:STDOUT:   %N.loc7_6.1: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic = %N.loc7_6.1 (constants.%N)]
// CHECK:STDOUT:   %array_type.loc7_42.1: type = array_type %N.loc7_6.1, constants.%C [symbolic = %array_type.loc7_42.1 (constants.%array_type.6a2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc7_42.1 [symbolic = %pattern_type (constants.%pattern_type.9ee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc7_42.1 [symbolic = %require_complete (constants.%require_complete.d82)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc7_6.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.818)]
// CHECK:STDOUT:   %bound_method.loc7_62.3: <bound method> = bound_method %N.loc7_6.1, constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc7_62.3 (constants.%bound_method.775)]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_62.2: init %i32 = call %bound_method.loc7_62.3(%N.loc7_6.1) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_62.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc7_42.1 (%array_type.6a2)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc7_61: Core.IntLiteral = name_ref N, %N.loc7_6.2 [symbolic = %N.loc7_6.1 (constants.%N)]
// CHECK:STDOUT:     %impl.elem0: %.1df = impl_witness_access constants.%ImplicitAs.impl_witness.204, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0f0]
// CHECK:STDOUT:     %bound_method.loc7_62.1: <bound method> = bound_method %N.ref.loc7_61, %impl.elem0 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.818)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_62.2: <bound method> = bound_method %N.ref.loc7_61, %specific_fn [symbolic = %bound_method.loc7_62.3 (constants.%bound_method.775)]
// CHECK:STDOUT:     %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_62.1: init %i32 = call %bound_method.loc7_62.2(%N.ref.loc7_61) [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_62.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     %.loc7_62: init %i32 = converted %N.ref.loc7_61, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_62.1 [symbolic = %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_62.2 (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:     return %.loc7_62 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.f30 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.f30 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.fe4 = var %a.var_patt
// CHECK:STDOUT:   %.loc11_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_35.1: %tuple.type.8d4 = tuple_literal (%.loc11_26.1, %.loc11_30.1, %.loc11_34.1)
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
// CHECK:STDOUT:   %.loc11_3.1: init %array_type.fe4 = converted %.loc11_35.1, %.loc11_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc11_3.1
// CHECK:STDOUT:   %.loc11_20: type = splice_block %array_type [concrete = constants.%array_type.fe4] {
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %D.ref [concrete = constants.%array_type.fe4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.fe4 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.fe4 = name_ref a, %a
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_3.1ba) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc22: %array_type.002 = converted %a.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(<error>)
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%array_type.fe4, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc11_3.2: %type_where = converted constants.%array_type.fe4, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.6f0
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.6f0, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.af6 = addr_of %a.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc7_6.1 => constants.%N
// CHECK:STDOUT:   %array_type.loc7_42.1 => constants.%array_type.6a2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ee
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_3.1ba) {
// CHECK:STDOUT:   %N.loc7_6.1 => constants.%int_3.1ba
// CHECK:STDOUT:   %array_type.loc7_42.1 => constants.%array_type.002
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a63
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.dd1
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound => constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.595
// CHECK:STDOUT:   %bound_method.loc7_62.3 => constants.%bound_method.f36
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc7_62.2 => constants.%int_3.822
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bound_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.c80: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7a9: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.71e: type = fn_type @ImplicitAs.Convert, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %From: Core.IntLiteral = bind_symbolic_name From, 0 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.543: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%From) [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.c08: %Int.as.ImplicitAs.impl.Convert.type.543 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bf: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e99, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.type.516: type = fn_type @Int.as.ImplicitAs.impl.Convert, @Int.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.b09: %Int.as.ImplicitAs.impl.Convert.type.516 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.7a9 = facet_value %i32, (%ImplicitAs.impl_witness.6bf) [concrete]
// CHECK:STDOUT:   %.81e: type = fn_type_with_self_type %ImplicitAs.Convert.type.71e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.51e, %Int.as.ImplicitAs.impl.Convert.b09 [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Int.as.ImplicitAs.impl.Convert.b09, @Int.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.b86: <bound method> = bound_method %N.51e, %Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.b86(%N.51e) [symbolic]
// CHECK:STDOUT:   %array_type.566: type = array_type %Int.as.ImplicitAs.impl.Convert.call, %C [symbolic]
// CHECK:STDOUT:   %pattern_type.9a8f: type = pattern_type %array_type.566 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.0fa: <witness> = require_complete_type %array_type.566 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.c80) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.51e, %Int.as.Copy.impl.Op.f59 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.8cf: <bound method> = bound_method %N.51e, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type.002: type = array_type %int_3, %C [concrete]
// CHECK:STDOUT:   %ptr.301: type = ptr_type %array_type.002 [concrete]
// CHECK:STDOUT:   %pattern_type.a63: type = pattern_type %array_type.002 [concrete]
// CHECK:STDOUT:   %tuple.type.8d4: type = tuple_type (%empty_struct_type, %empty_struct_type, %empty_struct_type) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %array: %array_type.002 = tuple_value (%C.val, %C.val, %C.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanAggregateDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %array_type.002, () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.type.0fd: type = fn_type @AggregateT.as_type.as.Destroy.impl.Op, @AggregateT.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.d02: %AggregateT.as_type.as.Destroy.impl.Op.type.0fd = struct_value () [concrete]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.25c: @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert.type (%Int.as.ImplicitAs.impl.Convert.type.543) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.ImplicitAs.impl.%Int.as.ImplicitAs.impl.Convert (constants.%Int.as.ImplicitAs.impl.Convert.c08)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e99 = impl_witness_table (%Core.import_ref.25c), @Int.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type (%pattern_type.9a8f) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type (%pattern_type.9a8f) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_34: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_34: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.1 (constants.%N.51e)]
// CHECK:STDOUT:     %a.param: @F.%array_type.loc6_28.1 (%array_type.566) = value_param call_param0
// CHECK:STDOUT:     %.loc6_28: type = splice_block %array_type.loc6_28.2 [symbolic = %array_type.loc6_28.1 (constants.%array_type.566)] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %N.ref.loc6_27: %i32 = name_ref N, %N.loc6_6.2 [symbolic = %N.loc6_6.1 (constants.%N.51e)]
// CHECK:STDOUT:       %impl.elem0.loc6_27: %.81e = impl_witness_access constants.%ImplicitAs.impl_witness.6bf, element0 [concrete = constants.%Int.as.ImplicitAs.impl.Convert.b09]
// CHECK:STDOUT:       %bound_method.loc6_27.2: <bound method> = bound_method %N.ref.loc6_27, %impl.elem0.loc6_27 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:       %specific_fn.loc6_27: <specific function> = specific_function %impl.elem0.loc6_27, @Int.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Int.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.loc6_27.3: <bound method> = bound_method %N.ref.loc6_27, %specific_fn.loc6_27 [symbolic = %bound_method.loc6_27.1 (constants.%bound_method.b86)]
// CHECK:STDOUT:       %Int.as.ImplicitAs.impl.Convert.call.loc6_27.2: init Core.IntLiteral = call %bound_method.loc6_27.3(%N.ref.loc6_27) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_27.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc6_27.1: Core.IntLiteral = value_of_initializer %Int.as.ImplicitAs.impl.Convert.call.loc6_27.2 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_27.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %.loc6_27.2: Core.IntLiteral = converted %N.ref.loc6_27, %.loc6_27.1 [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_27.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:       %array_type.loc6_28.2: type = array_type %.loc6_27.2, %C.ref [symbolic = %array_type.loc6_28.1 (constants.%array_type.566)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @F.%array_type.loc6_28.1 (%array_type.566) = bind_name a, %a.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT: generic fn @F(%N.loc6_6.2: %i32) {
// CHECK:STDOUT:   %N.loc6_6.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.1 (constants.%N.51e)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %N.loc6_6.1, constants.%Int.as.ImplicitAs.impl.Convert.b09 [symbolic = %Int.as.ImplicitAs.impl.Convert.bound (constants.%Int.as.ImplicitAs.impl.Convert.bound)]
// CHECK:STDOUT:   %bound_method.loc6_27.1: <bound method> = bound_method %N.loc6_6.1, constants.%Int.as.ImplicitAs.impl.Convert.specific_fn [symbolic = %bound_method.loc6_27.1 (constants.%bound_method.b86)]
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6_27.1: init Core.IntLiteral = call %bound_method.loc6_27.1(%N.loc6_6.1) [symbolic = %Int.as.ImplicitAs.impl.Convert.call.loc6_27.1 (constants.%Int.as.ImplicitAs.impl.Convert.call)]
// CHECK:STDOUT:   %array_type.loc6_28.1: type = array_type %Int.as.ImplicitAs.impl.Convert.call.loc6_27.1, constants.%C [symbolic = %array_type.loc6_28.1 (constants.%array_type.566)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %array_type.loc6_28.1 [symbolic = %pattern_type (constants.%pattern_type.9a8f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %array_type.loc6_28.1 [symbolic = %require_complete (constants.%require_complete.0fa)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc6_6.1, constants.%Int.as.Copy.impl.Op.f59 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:   %bound_method.loc6_47.3: <bound method> = bound_method %N.loc6_6.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_47.3 (constants.%bound_method.8cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @F.%array_type.loc6_28.1 (%array_type.566)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_47: %i32 = name_ref N, %N.loc6_6.2 [symbolic = %N.loc6_6.1 (constants.%N.51e)]
// CHECK:STDOUT:     %impl.elem0.loc6_47: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:     %bound_method.loc6_47.1: <bound method> = bound_method %N.ref.loc6_47, %impl.elem0.loc6_47 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound)]
// CHECK:STDOUT:     %specific_fn.loc6_47: <specific function> = specific_function %impl.elem0.loc6_47, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_47.2: <bound method> = bound_method %N.ref.loc6_47, %specific_fn.loc6_47 [symbolic = %bound_method.loc6_47.3 (constants.%bound_method.8cf)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_47.2(%N.ref.loc6_47) [symbolic = %N.loc6_6.1 (constants.%N.51e)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.a63 = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.a63 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %array_type.002 = var %a.var_patt
// CHECK:STDOUT:   %.loc9_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_34.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc9_35.1: %tuple.type.8d4 = tuple_literal (%.loc9_26.1, %.loc9_30.1, %.loc9_34.1)
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
// CHECK:STDOUT:   %.loc9_3.1: init %array_type.002 = converted %.loc9_35.1, %.loc9_35.8 [concrete = constants.%array]
// CHECK:STDOUT:   assign %a.var, %.loc9_3.1
// CHECK:STDOUT:   %.loc9_20: type = splice_block %array_type [concrete = constants.%array_type.002] {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3]
// CHECK:STDOUT:     %array_type: type = array_type %int_3, %C.ref [concrete = constants.%array_type.002]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %array_type.002 = bind_name a, %a.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: ref %array_type.002 = name_ref a, %a
// CHECK:STDOUT:   %facet_value: %type_where = facet_value constants.%array_type.002, () [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc9_3.2: %type_where = converted constants.%array_type.002, %facet_value [concrete = constants.%facet_value]
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %a.var, constants.%AggregateT.as_type.as.Destroy.impl.Op.d02
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%AggregateT.as_type.as.Destroy.impl.Op.d02, @AggregateT.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%AggregateT.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %a.var, %AggregateT.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.301 = addr_of %a.var
// CHECK:STDOUT:   %AggregateT.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return <error> to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc6_6.1 => constants.%N.51e
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.bound => constants.%Int.as.ImplicitAs.impl.Convert.bound
// CHECK:STDOUT:   %bound_method.loc6_27.1 => constants.%bound_method.b86
// CHECK:STDOUT:   %Int.as.ImplicitAs.impl.Convert.call.loc6_27.1 => constants.%Int.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %array_type.loc6_28.1 => constants.%array_type.566
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a8f
// CHECK:STDOUT: }
// CHECK:STDOUT:

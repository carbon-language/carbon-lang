// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/basics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/basics.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

class C;

//@dump-sem-ir-begin
fn A(p: const C**) -> const C** {
  return p;
}

fn B(p: const (C*)) -> const (C*) {
  return p;
}
//@dump-sem-ir-end

// --- collapse.carbon
library "[[@TEST_NAME]]";

class C;

// OK, `const (const C)` is the same type as `const C`.
//@dump-sem-ir-begin
// CHECK:STDERR: collapse.carbon:[[@LINE+4]]:23: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn F(p: const C**) -> const (const C)** {
// CHECK:STDERR:                       ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(p: const C**) -> const (const C)** {
  return p;
}
//@dump-sem-ir-end

// --- fail_collapse_in_error.carbon
library "[[@TEST_NAME]]";

class C;

// CHECK:STDERR: fail_collapse_in_error.carbon:[[@LINE+4]]:9: warning: `const` applied repeatedly to the same type has no additional effect [RepeatedConst]
// CHECK:STDERR: fn G(p: const (const C)**) -> C** {
// CHECK:STDERR:         ^~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: const (const C)**) -> C** {
  // CHECK:STDERR: fail_collapse_in_error.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `const C**` to `C**` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_collapse_in_error.carbon:[[@LINE+4]]:3: note: type `const C**` does not implement interface `Core.ImplicitAs(C**)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- add_or_remove_while_forming_value.carbon
library "[[@TEST_NAME]]";

class X {}

fn TakeValue(x: X) {}
fn TakeConstValue(x: const X) {}

fn PassConstValueToValue(a: const X) {
  TakeValue(a);
}

fn PassValueToConstValue(a: X) {
  TakeConstValue(a);
}

fn PassConstReferenceToValue(p: const X*) {
  TakeValue(*p);
}

fn PassReferenceToConstValue(p: X*) {
  TakeConstValue(*p);
}

// --- fail_todo_add_or_remove_while_initializing.carbon
library "[[@TEST_NAME]]";

class X {}

// TODO: None of these should require `X` to be copyable.
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+7]]:40: error: cannot copy value of type `X` [CopyOfUncopyableType]
// CHECK:STDERR: var init_non_const_from_non_const: X = {} as X;
// CHECK:STDERR:                                        ^~~~~~~
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+4]]:40: note: type `X` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var init_non_const_from_non_const: X = {} as X;
// CHECK:STDERR:                                        ^~~~~~~
// CHECK:STDERR:
var init_non_const_from_non_const: X = {} as X;
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+7]]:1: error: cannot copy value of type `X` [CopyOfUncopyableType]
// CHECK:STDERR: var init_non_const_from_const: X = ({} as X) as const X;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+4]]:1: note: type `X` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var init_non_const_from_const: X = ({} as X) as const X;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var init_non_const_from_const: X = ({} as X) as const X;
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+7]]:1: error: cannot copy value of type `const X` [CopyOfUncopyableType]
// CHECK:STDERR: var init_const_from_non_const: const X = {} as X;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+4]]:1: note: type `const X` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var init_const_from_non_const: const X = {} as X;
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var init_const_from_non_const: const X = {} as X;
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+7]]:38: error: cannot copy value of type `const X` [CopyOfUncopyableType]
// CHECK:STDERR: var init_const_from_const: const X = ({} as X) as const X;
// CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR: fail_todo_add_or_remove_while_initializing.carbon:[[@LINE+4]]:38: note: type `const X` does not implement interface `Core.Copy` [MissingImplInMemberAccessNote]
// CHECK:STDERR: var init_const_from_const: const X = ({} as X) as const X;
// CHECK:STDERR:                                      ^~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var init_const_from_const: const X = ({} as X) as const X;

// --- add_while_forming_reference.carbon
library "[[@TEST_NAME]]";

class X {
  fn TakeConstSelf[ref self: const Self]();
}

fn PassReferenceToConstReference(p: X*) {
  p->TakeConstSelf();
}

// --- fail_remove_while_forming_reference.carbon
library "[[@TEST_NAME]]";

class X {
  fn TakeSelf[ref self: Self]();
}

fn PassConstReferenceToReference(p: const X*) {
  // CHECK:STDERR: fail_remove_while_forming_reference.carbon:[[@LINE+10]]:3: error: cannot implicitly convert expression of type `const X` to `X` [ConversionFailure]
  // CHECK:STDERR:   p->(X.TakeSelf)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_remove_while_forming_reference.carbon:[[@LINE+7]]:3: note: type `const X` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   p->(X.TakeSelf)();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_remove_while_forming_reference.carbon:[[@LINE-10]]:15: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   fn TakeSelf[ref self: Self]();
  // CHECK:STDERR:               ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  p->(X.TakeSelf)();
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.0e5: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c45: type = ptr_type %const.0e5 [concrete]
// CHECK:STDOUT:   %ptr.728: type = ptr_type %ptr.c45 [concrete]
// CHECK:STDOUT:   %pattern_type.559: type = pattern_type %ptr.728 [concrete]
// CHECK:STDOUT:   %.c2e: form = init_form %ptr.728, call_param1 [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.5eb: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.035) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.cc6: %const.as.Copy.impl.Op.type.5eb = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.795: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%ptr.c45) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f8c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.c45) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f64: %ptr.as.Copy.impl.Op.type.f8c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.d1d: %Copy.type = facet_value %ptr.728, (%Copy.impl_witness.795) [concrete]
// CHECK:STDOUT:   %.de7: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.d1d [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.6a7: <specific function> = specific_function %ptr.as.Copy.impl.Op.f64, @ptr.as.Copy.impl.Op(%ptr.c45) [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %const.8ce: type = const_type %ptr.31e [concrete]
// CHECK:STDOUT:   %pattern_type.665: type = pattern_type %const.8ce [concrete]
// CHECK:STDOUT:   %.395: form = init_form %const.8ce, call_param1 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2c7: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %Copy.facet.a7f: %Copy.type = facet_value %ptr.31e, (%Copy.impl_witness.2c7) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.e93: <witness> = impl_witness imports.%Copy.impl_witness_table.a26, @const.as.Copy.impl(%Copy.facet.a7f) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.368: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.a7f) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.c19: %const.as.Copy.impl.Op.type.368 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.5cc: %Copy.type = facet_value %const.8ce, (%Copy.impl_witness.e93) [concrete]
// CHECK:STDOUT:   %.62c: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.5cc [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.c19, @const.as.Copy.impl.Op(%Copy.facet.a7f) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.ae5: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.5eb) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.cc6)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a26 = impl_witness_table (%Core.import_ref.ae5), @const.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.patt: %pattern_type.559 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.559 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.559 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.559 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc6_23: type = const_type %C.ref.loc6_29 [concrete = constants.%const.0e5]
// CHECK:STDOUT:     %ptr.loc6_30: type = ptr_type %const.loc6_23 [concrete = constants.%ptr.c45]
// CHECK:STDOUT:     %ptr.loc6_31: type = ptr_type %ptr.loc6_30 [concrete = constants.%ptr.728]
// CHECK:STDOUT:     %.loc6_31: form = init_form %ptr.loc6_31, call_param1 [concrete = constants.%.c2e]
// CHECK:STDOUT:     %p.param: %ptr.728 = value_param call_param0
// CHECK:STDOUT:     %.loc6_17: type = splice_block %ptr.loc6_17 [concrete = constants.%ptr.728] {
// CHECK:STDOUT:       %C.ref.loc6_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc6_9: type = const_type %C.ref.loc6_15 [concrete = constants.%const.0e5]
// CHECK:STDOUT:       %ptr.loc6_16: type = ptr_type %const.loc6_9 [concrete = constants.%ptr.c45]
// CHECK:STDOUT:       %ptr.loc6_17: type = ptr_type %ptr.loc6_16 [concrete = constants.%ptr.728]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.728 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.728 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.728 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %p.patt: %pattern_type.665 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.665 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.665 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.665 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc10_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_32: type = ptr_type %C.ref.loc10_31 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:     %const.loc10_24: type = const_type %ptr.loc10_32 [concrete = constants.%const.8ce]
// CHECK:STDOUT:     %.loc10_24: form = init_form %const.loc10_24, call_param1 [concrete = constants.%.395]
// CHECK:STDOUT:     %p.param: %const.8ce = value_param call_param0
// CHECK:STDOUT:     %.loc10_9: type = splice_block %const.loc10_9 [concrete = constants.%const.8ce] {
// CHECK:STDOUT:       %C.ref.loc10_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc10_17: type = ptr_type %C.ref.loc10_16 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:       %const.loc10_9: type = const_type %ptr.loc10_17 [concrete = constants.%const.8ce]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.8ce = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.8ce = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.8ce = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %ptr.728) -> out %return.param: %ptr.728 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.728 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.de7 = impl_witness_access constants.%Copy.impl_witness.795, element0 [concrete = constants.%ptr.as.Copy.impl.Op.f64]
// CHECK:STDOUT:   %bound_method.loc7_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.c45) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.6a7]
// CHECK:STDOUT:   %bound_method.loc7_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.728 = call %bound_method.loc7_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param: %const.8ce) -> out %return.param: %const.8ce {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.8ce = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.62c = impl_witness_access constants.%Copy.impl_witness.e93, element0 [concrete = constants.%const.as.Copy.impl.Op.c19]
// CHECK:STDOUT:   %bound_method.loc11_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.a7f) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.8ce = call %bound_method.loc11_10.2(%p.ref)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.0e5: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c45: type = ptr_type %const.0e5 [concrete]
// CHECK:STDOUT:   %ptr.728: type = ptr_type %ptr.c45 [concrete]
// CHECK:STDOUT:   %pattern_type.559: type = pattern_type %ptr.728 [concrete]
// CHECK:STDOUT:   %.c2e: form = init_form %ptr.728, call_param1 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.795: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%ptr.c45) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f8c: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.c45) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f64: %ptr.as.Copy.impl.Op.type.f8c = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.728, (%Copy.impl_witness.795) [concrete]
// CHECK:STDOUT:   %.de7: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.f64, @ptr.as.Copy.impl.Op(%ptr.c45) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %pattern_type.559 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.559 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.559 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.559 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc11_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc11_30: type = const_type %C.ref.loc11_36 [concrete = constants.%const.0e5]
// CHECK:STDOUT:     %const.loc11_23: type = const_type %const.loc11_30 [concrete = constants.%const.0e5]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %const.loc11_23 [concrete = constants.%ptr.c45]
// CHECK:STDOUT:     %ptr.loc11_39: type = ptr_type %ptr.loc11_38 [concrete = constants.%ptr.728]
// CHECK:STDOUT:     %.loc11_39: form = init_form %ptr.loc11_39, call_param1 [concrete = constants.%.c2e]
// CHECK:STDOUT:     %p.param: %ptr.728 = value_param call_param0
// CHECK:STDOUT:     %.loc11_17: type = splice_block %ptr.loc11_17 [concrete = constants.%ptr.728] {
// CHECK:STDOUT:       %C.ref.loc11_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc11_9: type = const_type %C.ref.loc11_15 [concrete = constants.%const.0e5]
// CHECK:STDOUT:       %ptr.loc11_16: type = ptr_type %const.loc11_9 [concrete = constants.%ptr.c45]
// CHECK:STDOUT:       %ptr.loc11_17: type = ptr_type %ptr.loc11_16 [concrete = constants.%ptr.728]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.728 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.728 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.728 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr.728) -> out %return.param: %ptr.728 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.728 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.de7 = impl_witness_access constants.%Copy.impl_witness.795, element0 [concrete = constants.%ptr.as.Copy.impl.Op.f64]
// CHECK:STDOUT:   %bound_method.loc12_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.c45) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.728 = call %bound_method.loc12_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:

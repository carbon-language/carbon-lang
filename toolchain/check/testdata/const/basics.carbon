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
  // CHECK:STDERR: fail_collapse_in_error.carbon:[[@LINE+4]]:3: note: type `const C**` does not implement interface `Core.ImplicitAs(C**)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// --- add_or_remove_while_forming_value.carbon
library "[[@TEST_NAME]]";

class X {}

fn TakeValue(unused x: X) {}
fn TakeConstValue(unused x: const X) {}

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

// --- add_or_remove_while_initializing.carbon
library "[[@TEST_NAME]]";

class X {}

var init_non_const_from_non_const: X = {} as X;
var init_non_const_from_const: X = ({} as X) as const X;
var init_const_from_non_const: const X = {} as X;
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
  // CHECK:STDERR: fail_remove_while_forming_reference.carbon:[[@LINE+7]]:3: note: type `const X` does not implement interface `Core.ImplicitAs(X)` [MissingImplInMemberAccessInContext]
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
// CHECK:STDOUT:   %const.a89: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.089: type = ptr_type %const.a89 [concrete]
// CHECK:STDOUT:   %ptr.826: type = ptr_type %ptr.089 [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %ptr.826 [concrete]
// CHECK:STDOUT:   %.77c: Core.Form = init_form %ptr.826 [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.6d8: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.f49) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.e95: %const.as.Copy.impl.Op.type.6d8 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ebf: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%ptr.089) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2c6: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.089) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.b55: %ptr.as.Copy.impl.Op.type.2c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.69b: %Copy.type = facet_value %ptr.826, (%Copy.impl_witness.ebf) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6ca: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.69b) [concrete]
// CHECK:STDOUT:   %.994: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6ca, %Copy.facet.69b [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.3a6: <specific function> = specific_function %ptr.as.Copy.impl.Op.b55, @ptr.as.Copy.impl.Op(%ptr.089) [concrete]
// CHECK:STDOUT:   %ptr.6be: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %const.ba2: type = const_type %ptr.6be [concrete]
// CHECK:STDOUT:   %pattern_type.0e9: type = pattern_type %const.ba2 [concrete]
// CHECK:STDOUT:   %.6ca: Core.Form = init_form %const.ba2 [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.881: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %Copy.facet.d5e: %Copy.type = facet_value %ptr.6be, (%Copy.impl_witness.881) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.ca1: <witness> = impl_witness imports.%Copy.impl_witness_table.b24, @const.as.Copy.impl(%Copy.facet.d5e) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.1a8: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.d5e) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.00c: %const.as.Copy.impl.Op.type.1a8 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.70c: %Copy.type = facet_value %const.ba2, (%Copy.impl_witness.ca1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.423: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.70c) [concrete]
// CHECK:STDOUT:   %.790: type = fn_type_with_self_type %Copy.WithSelf.Op.type.423, %Copy.facet.70c [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.00c, @const.as.Copy.impl.Op(%Copy.facet.d5e) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.60b: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.6d8) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.e95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b24 = impl_witness_table (%Core.import_ref.60b), @const.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.e85 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.e85 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e85 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e85 = return_slot_pattern %return.param_patt, %ptr.loc6_31 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc6_23: type = const_type %C.ref.loc6_29 [concrete = constants.%const.a89]
// CHECK:STDOUT:     %ptr.loc6_30: type = ptr_type %const.loc6_23 [concrete = constants.%ptr.089]
// CHECK:STDOUT:     %ptr.loc6_31: type = ptr_type %ptr.loc6_30 [concrete = constants.%ptr.826]
// CHECK:STDOUT:     %.loc6_31: Core.Form = init_form %ptr.loc6_31 [concrete = constants.%.77c]
// CHECK:STDOUT:     %p.param: %ptr.826 = value_param call_param0
// CHECK:STDOUT:     %.loc6_17: type = splice_block %ptr.loc6_17 [concrete = constants.%ptr.826] {
// CHECK:STDOUT:       %C.ref.loc6_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc6_9: type = const_type %C.ref.loc6_15 [concrete = constants.%const.a89]
// CHECK:STDOUT:       %ptr.loc6_16: type = ptr_type %const.loc6_9 [concrete = constants.%ptr.089]
// CHECK:STDOUT:       %ptr.loc6_17: type = ptr_type %ptr.loc6_16 [concrete = constants.%ptr.826]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.826 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.826 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.826 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.0e9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.0e9 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0e9 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.0e9 = return_slot_pattern %return.param_patt, %const.loc10_24 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc10_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_32: type = ptr_type %C.ref.loc10_31 [concrete = constants.%ptr.6be]
// CHECK:STDOUT:     %const.loc10_24: type = const_type %ptr.loc10_32 [concrete = constants.%const.ba2]
// CHECK:STDOUT:     %.loc10_24: Core.Form = init_form %const.loc10_24 [concrete = constants.%.6ca]
// CHECK:STDOUT:     %p.param: %const.ba2 = value_param call_param0
// CHECK:STDOUT:     %.loc10_9: type = splice_block %const.loc10_9 [concrete = constants.%const.ba2] {
// CHECK:STDOUT:       %C.ref.loc10_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc10_17: type = ptr_type %C.ref.loc10_16 [concrete = constants.%ptr.6be]
// CHECK:STDOUT:       %const.loc10_9: type = const_type %ptr.loc10_17 [concrete = constants.%const.ba2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.ba2 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.ba2 = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.ba2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %ptr.826) -> out %return.param: %ptr.826 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.826 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.994 = impl_witness_access constants.%Copy.impl_witness.ebf, element0 [concrete = constants.%ptr.as.Copy.impl.Op.b55]
// CHECK:STDOUT:   %bound_method.loc7_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.089) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.3a6]
// CHECK:STDOUT:   %bound_method.loc7_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.826 = call %bound_method.loc7_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param: %const.ba2) -> out %return.param: %const.ba2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.ba2 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.790 = impl_witness_access constants.%Copy.impl_witness.ca1, element0 [concrete = constants.%const.as.Copy.impl.Op.00c]
// CHECK:STDOUT:   %bound_method.loc11_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.d5e) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.ba2 = call %bound_method.loc11_10.2(%p.ref)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.a89: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.089: type = ptr_type %const.a89 [concrete]
// CHECK:STDOUT:   %ptr.826: type = ptr_type %ptr.089 [concrete]
// CHECK:STDOUT:   %pattern_type.e85: type = pattern_type %ptr.826 [concrete]
// CHECK:STDOUT:   %.77c: Core.Form = init_form %ptr.826 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.aa4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.5a1: %ptr.as.Copy.impl.Op.type.aa4 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.ebf: <witness> = impl_witness imports.%Copy.impl_witness_table.6b9, @ptr.as.Copy.impl(%ptr.089) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2c6: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.089) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.b55: %ptr.as.Copy.impl.Op.type.2c6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.826, (%Copy.impl_witness.ebf) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6ca: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.994: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6ca, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.b55, @ptr.as.Copy.impl.Op(%ptr.089) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.aa4: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.aa4) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.5a1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.6b9 = impl_witness_table (%Core.import_ref.aa4), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.e85 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.e85 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e85 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e85 = return_slot_pattern %return.param_patt, %ptr.loc11_39 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc11_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc11_30: type = const_type %C.ref.loc11_36 [concrete = constants.%const.a89]
// CHECK:STDOUT:     %const.loc11_23: type = const_type %const.loc11_30 [concrete = constants.%const.a89]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %const.loc11_23 [concrete = constants.%ptr.089]
// CHECK:STDOUT:     %ptr.loc11_39: type = ptr_type %ptr.loc11_38 [concrete = constants.%ptr.826]
// CHECK:STDOUT:     %.loc11_39: Core.Form = init_form %ptr.loc11_39 [concrete = constants.%.77c]
// CHECK:STDOUT:     %p.param: %ptr.826 = value_param call_param0
// CHECK:STDOUT:     %.loc11_17: type = splice_block %ptr.loc11_17 [concrete = constants.%ptr.826] {
// CHECK:STDOUT:       %C.ref.loc11_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc11_9: type = const_type %C.ref.loc11_15 [concrete = constants.%const.a89]
// CHECK:STDOUT:       %ptr.loc11_16: type = ptr_type %const.loc11_9 [concrete = constants.%ptr.089]
// CHECK:STDOUT:       %ptr.loc11_17: type = ptr_type %ptr.loc11_16 [concrete = constants.%ptr.826]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.826 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.826 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.826 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr.826) -> out %return.param: %ptr.826 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.826 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.994 = impl_witness_access constants.%Copy.impl_witness.ebf, element0 [concrete = constants.%ptr.as.Copy.impl.Op.b55]
// CHECK:STDOUT:   %bound_method.loc12_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.089) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.826 = call %bound_method.loc12_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:

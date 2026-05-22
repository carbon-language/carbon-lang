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
// CHECK:STDOUT:   %const.ec2: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.e14: type = ptr_type %const.ec2 [concrete]
// CHECK:STDOUT:   %ptr.675: type = ptr_type %ptr.e14 [concrete]
// CHECK:STDOUT:   %pattern_type.b09: type = pattern_type %ptr.675 [concrete]
// CHECK:STDOUT:   %.4e2: Core.Form = init_form %ptr.675 [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.228: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.545: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%T.228) [symbolic]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.376: %const.as.Copy.impl.Op.type.545 = struct_value () [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.c81: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%ptr.e14) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.a4d: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.e14) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.c12: %ptr.as.Copy.impl.Op.type.a4d = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.9aa: %Copy.type = facet_value %ptr.675, (%Copy.impl_witness.c81) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.455: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.9aa) [concrete]
// CHECK:STDOUT:   %.f9c: type = fn_type_with_self_type %Copy.WithSelf.Op.type.455, %Copy.facet.9aa [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn.752: <specific function> = specific_function %ptr.as.Copy.impl.Op.c12, @ptr.as.Copy.impl.Op(%ptr.e14) [concrete]
// CHECK:STDOUT:   %ptr.4af: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %const.eea: type = const_type %ptr.4af [concrete]
// CHECK:STDOUT:   %pattern_type.5d4: type = pattern_type %const.eea [concrete]
// CHECK:STDOUT:   %.6b1: Core.Form = init_form %const.eea [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.805: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %Copy.facet.457: %Copy.type = facet_value %ptr.4af, (%Copy.impl_witness.805) [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6b2: <witness> = impl_witness imports.%Copy.impl_witness_table.032, @const.as.Copy.impl(%Copy.facet.457) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.type.ccc: type = fn_type @const.as.Copy.impl.Op, @const.as.Copy.impl(%Copy.facet.457) [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.632: %const.as.Copy.impl.Op.type.ccc = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.13b: %Copy.type = facet_value %const.eea, (%Copy.impl_witness.6b2) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.2c8: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.13b) [concrete]
// CHECK:STDOUT:   %.ee3: type = fn_type_with_self_type %Copy.WithSelf.Op.type.2c8, %Copy.facet.13b [concrete]
// CHECK:STDOUT:   %const.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %const.as.Copy.impl.Op.632, @const.as.Copy.impl.Op(%Copy.facet.457) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.583: @const.as.Copy.impl.%const.as.Copy.impl.Op.type (%const.as.Copy.impl.Op.type.545) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @const.as.Copy.impl.%const.as.Copy.impl.Op (constants.%const.as.Copy.impl.Op.376)]
// CHECK:STDOUT:   %Copy.impl_witness_table.032 = impl_witness_table (%Core.import_ref.583), @const.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.b09 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.b09 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b09 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.b09 = return_slot_pattern %return.param_patt, %ptr.loc6_31 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc6_29: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc6_23: type = const_type %C.ref.loc6_29 [concrete = constants.%const.ec2]
// CHECK:STDOUT:     %ptr.loc6_30: type = ptr_type %const.loc6_23 [concrete = constants.%ptr.e14]
// CHECK:STDOUT:     %ptr.loc6_31: type = ptr_type %ptr.loc6_30 [concrete = constants.%ptr.675]
// CHECK:STDOUT:     %.loc6_31: Core.Form = init_form %ptr.loc6_31 [concrete = constants.%.4e2]
// CHECK:STDOUT:     %p.param: %ptr.675 = value_param call_param0
// CHECK:STDOUT:     %.loc6_17: type = splice_block %ptr.loc6_17 [concrete = constants.%ptr.675] {
// CHECK:STDOUT:       %C.ref.loc6_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc6_9: type = const_type %C.ref.loc6_15 [concrete = constants.%const.ec2]
// CHECK:STDOUT:       %ptr.loc6_16: type = ptr_type %const.loc6_9 [concrete = constants.%ptr.e14]
// CHECK:STDOUT:       %ptr.loc6_17: type = ptr_type %ptr.loc6_16 [concrete = constants.%ptr.675]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.675 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.675 = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.675 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.5d4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.5d4 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5d4 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5d4 = return_slot_pattern %return.param_patt, %const.loc10_24 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc10_31: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %ptr.loc10_32: type = ptr_type %C.ref.loc10_31 [concrete = constants.%ptr.4af]
// CHECK:STDOUT:     %const.loc10_24: type = const_type %ptr.loc10_32 [concrete = constants.%const.eea]
// CHECK:STDOUT:     %.loc10_24: Core.Form = init_form %const.loc10_24 [concrete = constants.%.6b1]
// CHECK:STDOUT:     %p.param: %const.eea = value_param call_param0
// CHECK:STDOUT:     %.loc10_9: type = splice_block %const.loc10_9 [concrete = constants.%const.eea] {
// CHECK:STDOUT:       %C.ref.loc10_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr.loc10_17: type = ptr_type %C.ref.loc10_16 [concrete = constants.%ptr.4af]
// CHECK:STDOUT:       %const.loc10_9: type = const_type %ptr.loc10_17 [concrete = constants.%const.eea]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %const.eea = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.eea = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.eea = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %ptr.675) -> out %return.param: %ptr.675 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.675 = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.f9c = impl_witness_access constants.%Copy.impl_witness.c81, element0 [concrete = constants.%ptr.as.Copy.impl.Op.c12]
// CHECK:STDOUT:   %bound_method.loc7_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.e14) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn.752]
// CHECK:STDOUT:   %bound_method.loc7_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.675 = call %bound_method.loc7_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param: %const.eea) -> out %return.param: %const.eea {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %const.eea = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.ee3 = impl_witness_access constants.%Copy.impl_witness.6b2, element0 [concrete = constants.%const.as.Copy.impl.Op.632]
// CHECK:STDOUT:   %bound_method.loc11_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @const.as.Copy.impl.Op(constants.%Copy.facet.457) [concrete = constants.%const.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc11_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %const.as.Copy.impl.Op.call: init %const.eea = call %bound_method.loc11_10.2(%p.ref)
// CHECK:STDOUT:   return %const.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.ebe: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.ef2: type = ptr_type %const.ebe [concrete]
// CHECK:STDOUT:   %ptr.98f: type = ptr_type %ptr.ef2 [concrete]
// CHECK:STDOUT:   %pattern_type.2a8: type = pattern_type %ptr.98f [concrete]
// CHECK:STDOUT:   %.42a: Core.Form = init_form %ptr.98f [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b03: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.aed: %ptr.as.Copy.impl.Op.type.b03 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f46: <witness> = impl_witness imports.%Copy.impl_witness_table.760, @ptr.as.Copy.impl(%ptr.ef2) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.328: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%ptr.ef2) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.430: %ptr.as.Copy.impl.Op.type.328 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.98f, (%Copy.impl_witness.f46) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f90: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a65: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f90, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.430, @ptr.as.Copy.impl.Op(%ptr.ef2) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.13c: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.b03) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.aed)]
// CHECK:STDOUT:   %Copy.impl_witness_table.760 = impl_witness_table (%Core.import_ref.13c), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.2a8 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.2a8 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.2a8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.2a8 = return_slot_pattern %return.param_patt, %ptr.loc11_39 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc11_36: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc11_30: type = const_type %C.ref.loc11_36 [concrete = constants.%const.ebe]
// CHECK:STDOUT:     %const.loc11_23: type = const_type %const.loc11_30 [concrete = constants.%const.ebe]
// CHECK:STDOUT:     %ptr.loc11_38: type = ptr_type %const.loc11_23 [concrete = constants.%ptr.ef2]
// CHECK:STDOUT:     %ptr.loc11_39: type = ptr_type %ptr.loc11_38 [concrete = constants.%ptr.98f]
// CHECK:STDOUT:     %.loc11_39: Core.Form = init_form %ptr.loc11_39 [concrete = constants.%.42a]
// CHECK:STDOUT:     %p.param: %ptr.98f = value_param call_param0
// CHECK:STDOUT:     %.loc11_17: type = splice_block %ptr.loc11_17 [concrete = constants.%ptr.98f] {
// CHECK:STDOUT:       %C.ref.loc11_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc11_9: type = const_type %C.ref.loc11_15 [concrete = constants.%const.ebe]
// CHECK:STDOUT:       %ptr.loc11_16: type = ptr_type %const.loc11_9 [concrete = constants.%ptr.ef2]
// CHECK:STDOUT:       %ptr.loc11_17: type = ptr_type %ptr.loc11_16 [concrete = constants.%ptr.98f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.98f = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %ptr.98f = out_param call_param1
// CHECK:STDOUT:     %return: ref %ptr.98f = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr.98f) -> out %return.param: %ptr.98f {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr.98f = name_ref p, %p
// CHECK:STDOUT:   %impl.elem0: %.a65 = impl_witness_access constants.%Copy.impl_witness.f46, element0 [concrete = constants.%ptr.as.Copy.impl.Op.430]
// CHECK:STDOUT:   %bound_method.loc12_10.1: <bound method> = bound_method %p.ref, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @ptr.as.Copy.impl.Op(constants.%ptr.ef2) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_10.2: <bound method> = bound_method %p.ref, %specific_fn
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call: init %ptr.98f = call %bound_method.loc12_10.2(%p.ref)
// CHECK:STDOUT:   return %ptr.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:

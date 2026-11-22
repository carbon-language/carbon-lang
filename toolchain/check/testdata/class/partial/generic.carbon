// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/partial/generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/partial/generic.carbon

// --- base_class.carbon
library "[[@TEST_NAME]]";

base class B {}

//@dump-sem-ir-begin
fn F[T: Core.Extendable](unused p: partial T*) {}
//@dump-sem-ir-end

fn G(b: partial B*) {
  //@dump-sem-ir-begin
  F(b);
  //@dump-sem-ir-end
}

// --- abstract_class.carbon
library "[[@TEST_NAME]]";

abstract class A {}

//@dump-sem-ir-begin
fn F[T: Core.Extendable](unused p: partial T*) {}
//@dump-sem-ir-end

fn G(a: partial A*) {
  //@dump-sem-ir-begin
  F(a);
  //@dump-sem-ir-end
}

// --- convert.carbon
library "[[@TEST_NAME]]";

base class B {}

//@dump-sem-ir-begin
fn F[T: Core.Extendable](ref t: T) -> partial T* {
  return &(t as partial T);
}
//@dump-sem-ir-end

fn Test(ref b: B) {
  //@dump-sem-ir-begin
  F(ref b);
  //@dump-sem-ir-end
}

// --- period_self.carbon
library "[[@TEST_NAME]]";

interface I {
  fn Op(ref self);
}

base class B {}

impl partial B as I {
  fn Op(ref unused self) {}
}

fn F[T: Core.Extendable where partial .Self impls I](ref t: T) {
  (t as partial T).(I.Op)();
}

fn G(ref b: B) {
  F(ref b);
}

// --- fail_period_self_unconstrained.carbon
library "[[@TEST_NAME]]";

interface I {}

// CHECK:STDERR:        ╭─┤ fail_period_self_unconstrained.carbon:[[@LINE+8]]:27
// CHECK:STDERR: ->     │ fn F[unused T: type where partial .Self impls I]() {}
// CHECK:STDERR:        ·                           ┳━━━━━━━━━━━━
// CHECK:STDERR:        ·                           ┗━━┫ error: `partial` applied to type `.Self` which
// CHECK:STDERR:        ·                              ┃        does not implement `Core.Extendable`
// CHECK:STDERR:        ·                              ┃        [PartialOnNonExtendable]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn F[unused T: type where partial .Self impls I]() {}

// --- fail_unconstrained_member_access.carbon
library "[[@TEST_NAME]]";

interface I {
  fn Op(ref self);
}

fn F[T: Core.Extendable](ref t: T) {
  // CHECK:STDERR:        ╭─┤ fail_unconstrained_member_access.carbon:[[@LINE+8]]:3
  // CHECK:STDERR: ->     │   (t as partial T).(I.Op)();
  // CHECK:STDERR:        ·   ┳━━━━━━━━━━━━━━━━━━━━━━
  // CHECK:STDERR:        ·   ┗━━┫ error: cannot access member of interface `I` in type `partial T` that
  // CHECK:STDERR:        ·      ┃        does not implement that interface
  // CHECK:STDERR:        ·      ┃        [MissingImplInMemberAccess]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  (t as partial T).(I.Op)();
}

// --- fail_final_class.carbon
library "[[@TEST_NAME]]";

class C {}

fn F[T: Core.Extendable](unused t: T) {}

fn G(c: C) {
  // CHECK:STDERR:        ╭─┤ fail_final_class.carbon:[[@LINE+12]]:3
  // CHECK:STDERR:        │ fn F[T: Core.Extendable](unused t: T) {}
  // CHECK:STDERR:        ·    ┬
  // CHECK:STDERR:        ·    ╰──┤ while deducing parameters of generic declared here
  // CHECK:STDERR:        ·       │ [DeductionGenericHere]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(c);
  // CHECK:STDERR:        ·   ━━┳━
  // CHECK:STDERR:        ·     ┗━━┫ error: cannot convert type `C` into type implementing `Core.Extendable`
  // CHECK:STDERR:        ·        ┃        [ConversionFailureTypeToFacet]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(c);
}

// CHECK:STDOUT: --- base_class.carbon = $base_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Extendable.type: type = facet_type <@Extendable> [concrete]
// CHECK:STDOUT:   %pattern_type.7e1: type = pattern_type %Extendable.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Extendable.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %.437: type = partial_type %T.as_type [symbolic]
// CHECK:STDOUT:   %ptr.154: type = ptr_type %.437 [symbolic]
// CHECK:STDOUT:   %pattern_type.b5e: type = pattern_type %ptr.154 [symbolic]
// CHECK:STDOUT:   %p.param_patt.a66: %pattern_type.b5e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.7ae: %pattern_type.b5e = wrapper_binding_pattern p, %p.param_patt.a66 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.154 [symbolic]
// CHECK:STDOUT:   %.640: type = partial_type %B [concrete]
// CHECK:STDOUT:   %ptr.3dc: type = ptr_type %.640 [concrete]
// CHECK:STDOUT:   %pattern_type.777: type = pattern_type %ptr.3dc [concrete]
// CHECK:STDOUT:   %custom_witness: <witness> = custom_witness (), @Extendable [concrete]
// CHECK:STDOUT:   %Extendable.facet: %Extendable.type = facet_value %B, (%custom_witness) [concrete]
// CHECK:STDOUT:   %p.param_patt.4aa: %pattern_type.777 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.3ac: %pattern_type.777 = wrapper_binding_pattern p, %p.param_patt.4aa [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Extendable.facet) [concrete]
// CHECK:STDOUT:   %complete_type.df9: <witness> = complete_type_witness %ptr.3dc [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Extendable = %Core.Extendable
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Extendable: type = import_ref Core//prelude/parts/destroy, Extendable, loaded [concrete = constants.%Extendable.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.param_patt.loc6_34.1: @F.%pattern_type (%pattern_type.b5e) = value_param_pattern [symbolic = %p.param_patt.loc6_34.2 (constants.%p.param_patt.a66)]
// CHECK:STDOUT:     %p.patt.loc6_34.1: @F.%pattern_type (%pattern_type.b5e) = wrapper_binding_pattern p, %p.param_patt.loc6_34.1 [symbolic = %p.patt.loc6_34.2 (constants.%p.patt.7ae)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_13: type = splice_block %Extendable.ref [concrete = constants.%Extendable.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Extendable.ref: type = name_ref Extendable, imports.%Core.Extendable [concrete = constants.%Extendable.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %Extendable.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_45.1 (%ptr.154) = value_param call_param0
// CHECK:STDOUT:     %.loc6_45: type = splice_block %ptr.loc6_45.2 [symbolic = %ptr.loc6_45.1 (constants.%ptr.154)] {
// CHECK:STDOUT:       %T.ref: %Extendable.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6_36.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_36.2: type = converted %T.ref, %T.as_type.loc6_36.2 [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_36.3: type = partial_type %.loc6_36.2 [symbolic = %.loc6_36.1 (constants.%.437)]
// CHECK:STDOUT:       %ptr.loc6_45.2: type = ptr_type %.loc6_36.3 [symbolic = %ptr.loc6_45.1 (constants.%ptr.154)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_45.1 (%ptr.154) = wrapper_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: %Extendable.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: %Extendable.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc6_36.1: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc6_36.1: type = partial_type %T.as_type.loc6_36.1 [symbolic = %.loc6_36.1 (constants.%.437)]
// CHECK:STDOUT:   %ptr.loc6_45.1: type = ptr_type %.loc6_36.1 [symbolic = %ptr.loc6_45.1 (constants.%ptr.154)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc6_45.1 [symbolic = %pattern_type (constants.%pattern_type.b5e)]
// CHECK:STDOUT:   %p.param_patt.loc6_34.2: @F.%pattern_type (%pattern_type.b5e) = value_param_pattern [symbolic = %p.param_patt.loc6_34.2 (constants.%p.param_patt.a66)]
// CHECK:STDOUT:   %p.patt.loc6_34.2: @F.%pattern_type (%pattern_type.b5e) = wrapper_binding_pattern p, %p.param_patt.loc6_34.2 [symbolic = %p.patt.loc6_34.2 (constants.%p.patt.7ae)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc6_45.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_45.1 (%ptr.154)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%b.param: %ptr.3dc) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %b.ref: %ptr.3dc = name_ref b, %b
// CHECK:STDOUT:   %Extendable.facet.loc11_6.1: %Extendable.type = facet_value constants.%B, (constants.%custom_witness) [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %.loc11_6.1: %Extendable.type = converted constants.%B, %Extendable.facet.loc11_6.1 [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %Extendable.facet.loc11_6.2: %Extendable.type = facet_value constants.%B, (constants.%custom_witness) [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %.loc11_6.2: %Extendable.type = converted constants.%B, %Extendable.facet.loc11_6.2 [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Extendable.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc6_36.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc6_36.1 => constants.%.437
// CHECK:STDOUT:   %ptr.loc6_45.1 => constants.%ptr.154
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b5e
// CHECK:STDOUT:   %p.param_patt.loc6_34.2 => constants.%p.param_patt.a66
// CHECK:STDOUT:   %p.patt.loc6_34.2 => constants.%p.patt.7ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Extendable.facet) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%Extendable.facet
// CHECK:STDOUT:   %T.as_type.loc6_36.1 => constants.%B
// CHECK:STDOUT:   %.loc6_36.1 => constants.%.640
// CHECK:STDOUT:   %ptr.loc6_45.1 => constants.%ptr.3dc
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.777
// CHECK:STDOUT:   %p.param_patt.loc6_34.2 => constants.%p.param_patt.4aa
// CHECK:STDOUT:   %p.patt.loc6_34.2 => constants.%p.patt.3ac
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.df9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_class.carbon = $abstract_class
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Extendable.type: type = facet_type <@Extendable> [concrete]
// CHECK:STDOUT:   %pattern_type.7e1: type = pattern_type %Extendable.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: %Extendable.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [symbolic]
// CHECK:STDOUT:   %.437: type = partial_type %T.as_type [symbolic]
// CHECK:STDOUT:   %ptr.154: type = ptr_type %.437 [symbolic]
// CHECK:STDOUT:   %pattern_type.b5e: type = pattern_type %ptr.154 [symbolic]
// CHECK:STDOUT:   %p.param_patt.a66: %pattern_type.b5e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.7ae: %pattern_type.b5e = wrapper_binding_pattern p, %p.param_patt.a66 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.154 [symbolic]
// CHECK:STDOUT:   %.3d6: type = partial_type %A [concrete]
// CHECK:STDOUT:   %ptr.f01: type = ptr_type %.3d6 [concrete]
// CHECK:STDOUT:   %pattern_type.758: type = pattern_type %ptr.f01 [concrete]
// CHECK:STDOUT:   %custom_witness: <witness> = custom_witness (), @Extendable [concrete]
// CHECK:STDOUT:   %Extendable.facet: %Extendable.type = facet_value %A, (%custom_witness) [concrete]
// CHECK:STDOUT:   %p.param_patt.c4b: %pattern_type.758 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.b35: %pattern_type.758 = wrapper_binding_pattern p, %p.param_patt.c4b [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Extendable.facet) [concrete]
// CHECK:STDOUT:   %complete_type.485: <witness> = complete_type_witness %ptr.f01 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Extendable = %Core.Extendable
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Extendable: type = import_ref Core//prelude/parts/destroy, Extendable, loaded [concrete = constants.%Extendable.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.param_patt.loc6_34.1: @F.%pattern_type (%pattern_type.b5e) = value_param_pattern [symbolic = %p.param_patt.loc6_34.2 (constants.%p.param_patt.a66)]
// CHECK:STDOUT:     %p.patt.loc6_34.1: @F.%pattern_type (%pattern_type.b5e) = wrapper_binding_pattern p, %p.param_patt.loc6_34.1 [symbolic = %p.patt.loc6_34.2 (constants.%p.patt.7ae)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_13: type = splice_block %Extendable.ref [concrete = constants.%Extendable.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Extendable.ref: type = name_ref Extendable, imports.%Core.Extendable [concrete = constants.%Extendable.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %Extendable.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_45.1 (%ptr.154) = value_param call_param0
// CHECK:STDOUT:     %.loc6_45: type = splice_block %ptr.loc6_45.2 [symbolic = %ptr.loc6_45.1 (constants.%ptr.154)] {
// CHECK:STDOUT:       %T.ref: %Extendable.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6_36.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_36.2: type = converted %T.ref, %T.as_type.loc6_36.2 [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_36.3: type = partial_type %.loc6_36.2 [symbolic = %.loc6_36.1 (constants.%.437)]
// CHECK:STDOUT:       %ptr.loc6_45.2: type = ptr_type %.loc6_36.3 [symbolic = %ptr.loc6_45.1 (constants.%ptr.154)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_45.1 (%ptr.154) = wrapper_binding p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: %Extendable.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc6_7.1: %Extendable.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc6_36.1: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc6_36.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc6_36.1: type = partial_type %T.as_type.loc6_36.1 [symbolic = %.loc6_36.1 (constants.%.437)]
// CHECK:STDOUT:   %ptr.loc6_45.1: type = ptr_type %.loc6_36.1 [symbolic = %ptr.loc6_45.1 (constants.%ptr.154)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc6_45.1 [symbolic = %pattern_type (constants.%pattern_type.b5e)]
// CHECK:STDOUT:   %p.param_patt.loc6_34.2: @F.%pattern_type (%pattern_type.b5e) = value_param_pattern [symbolic = %p.param_patt.loc6_34.2 (constants.%p.param_patt.a66)]
// CHECK:STDOUT:   %p.patt.loc6_34.2: @F.%pattern_type (%pattern_type.b5e) = wrapper_binding_pattern p, %p.param_patt.loc6_34.2 [symbolic = %p.patt.loc6_34.2 (constants.%p.patt.7ae)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc6_45.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_45.1 (%ptr.154)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%a.param: %ptr.f01) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %a.ref: %ptr.f01 = name_ref a, %a
// CHECK:STDOUT:   %Extendable.facet.loc11_6.1: %Extendable.type = facet_value constants.%A, (constants.%custom_witness) [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %.loc11_6.1: %Extendable.type = converted constants.%A, %Extendable.facet.loc11_6.1 [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %Extendable.facet.loc11_6.2: %Extendable.type = facet_value constants.%A, (constants.%custom_witness) [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %.loc11_6.2: %Extendable.type = converted constants.%A, %Extendable.facet.loc11_6.2 [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Extendable.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%a.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc6_36.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc6_36.1 => constants.%.437
// CHECK:STDOUT:   %ptr.loc6_45.1 => constants.%ptr.154
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b5e
// CHECK:STDOUT:   %p.param_patt.loc6_34.2 => constants.%p.param_patt.a66
// CHECK:STDOUT:   %p.patt.loc6_34.2 => constants.%p.patt.7ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Extendable.facet) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%Extendable.facet
// CHECK:STDOUT:   %T.as_type.loc6_36.1 => constants.%A
// CHECK:STDOUT:   %.loc6_36.1 => constants.%.3d6
// CHECK:STDOUT:   %ptr.loc6_45.1 => constants.%ptr.f01
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.758
// CHECK:STDOUT:   %p.param_patt.loc6_34.2 => constants.%p.param_patt.c4b
// CHECK:STDOUT:   %p.patt.loc6_34.2 => constants.%p.patt.b35
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.485
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- convert.carbon = $convert.2
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Extendable.type: type = facet_type <@Extendable> [concrete]
// CHECK:STDOUT:   %pattern_type.7e1: type = pattern_type %Extendable.type [concrete]
// CHECK:STDOUT:   %T.patt.993: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.44d: %Extendable.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.43e: type = facet_access_type %T.44d [symbolic]
// CHECK:STDOUT:   %pattern_type.7a5: type = pattern_type %T.as_type.43e [symbolic]
// CHECK:STDOUT:   %t.param_patt.0cf: %pattern_type.7a5 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %t.patt.cb0: %pattern_type.7a5 = wrapper_binding_pattern t, %t.param_patt.0cf [symbolic]
// CHECK:STDOUT:   %.437: type = partial_type %T.as_type.43e [symbolic]
// CHECK:STDOUT:   %ptr.154: type = ptr_type %.437 [symbolic]
// CHECK:STDOUT:   %.970: Core.Form = init_form %ptr.154 [symbolic]
// CHECK:STDOUT:   %pattern_type.b5e: type = pattern_type %ptr.154 [symbolic]
// CHECK:STDOUT:   %return.param_patt.2f0: %pattern_type.b5e = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.8e6: %pattern_type.b5e = return_slot_pattern %return.param_patt.2f0, %ptr.154 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.b36: <witness> = require_complete_type %T.as_type.43e [symbolic]
// CHECK:STDOUT:   %require_complete.f97: <witness> = require_complete_type %ptr.154 [symbolic]
// CHECK:STDOUT:   %require_complete.ed9: <witness> = require_complete_type %.437 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.20f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.f12: %ptr.as.Copy.impl.Op.type.20f = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.858: <witness> = lookup_impl_witness %ptr.154, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.fb9: %Copy.type = facet_value %ptr.154, (%Copy.lookup_impl_witness.858) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.75d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.fb9) [symbolic]
// CHECK:STDOUT:   %.ae6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.75d, %Copy.facet.fb9 [symbolic]
// CHECK:STDOUT:   %impl.elem0.f56: %.ae6 = impl_witness_access %Copy.lookup_impl_witness.858, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e6e: <specific function> = specific_impl_function %impl.elem0.f56, @Copy.WithSelf.Op(%Copy.facet.fb9) [symbolic]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %custom_witness: <witness> = custom_witness (), @Extendable [concrete]
// CHECK:STDOUT:   %Extendable.facet: %Extendable.type = facet_value %B, (%custom_witness) [concrete]
// CHECK:STDOUT:   %t.param_patt.375: %pattern_type.e39 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %t.patt.cab: %pattern_type.e39 = wrapper_binding_pattern t, %t.param_patt.375 [concrete]
// CHECK:STDOUT:   %.640: type = partial_type %B [concrete]
// CHECK:STDOUT:   %ptr.3dc: type = ptr_type %.640 [concrete]
// CHECK:STDOUT:   %.3c5: Core.Form = init_form %ptr.3dc [concrete]
// CHECK:STDOUT:   %pattern_type.777: type = pattern_type %ptr.3dc [concrete]
// CHECK:STDOUT:   %return.param_patt.e41: %pattern_type.777 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.7d3: %pattern_type.777 = return_slot_pattern %return.param_patt.e41, %ptr.3dc [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%Extendable.facet) [concrete]
// CHECK:STDOUT:   %complete_type.df9: <witness> = complete_type_witness %ptr.3dc [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.45a: <witness> = impl_witness imports.%Copy.impl_witness_table.fb0, @ptr.as.Copy.impl(%.640) [concrete]
// CHECK:STDOUT:   %Copy.facet.8c6: %Copy.type = facet_value %ptr.3dc, (%Copy.impl_witness.45a) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.b75: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.8c6) [concrete]
// CHECK:STDOUT:   %.a42: type = fn_type_with_self_type %Copy.WithSelf.Op.type.b75, %Copy.facet.8c6 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.db6: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%.640) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.652: %ptr.as.Copy.impl.Op.type.db6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.652, @ptr.as.Copy.impl.Op(%.640) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Extendable = %Core.Extendable
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Extendable: type = import_ref Core//prelude/parts/destroy, Extendable, loaded [concrete = constants.%Extendable.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.06a: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.20f) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.f12)]
// CHECK:STDOUT:   %Copy.impl_witness_table.fb0 = impl_witness_table (%Core.import_ref.06a), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc6_7.1: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt.993)]
// CHECK:STDOUT:     %t.param_patt.loc6_31.1: @F.%pattern_type.loc6_31 (%pattern_type.7a5) = ref_param_pattern [symbolic = %t.param_patt.loc6_31.2 (constants.%t.param_patt.0cf)]
// CHECK:STDOUT:     %t.patt.loc6_31.1: @F.%pattern_type.loc6_31 (%pattern_type.7a5) = wrapper_binding_pattern t, %t.param_patt.loc6_31.1 [symbolic = %t.patt.loc6_31.2 (constants.%t.patt.cb0)]
// CHECK:STDOUT:     %return.param_patt.loc6_48.1: @F.%pattern_type.loc6_48 (%pattern_type.b5e) = out_param_pattern [symbolic = %return.param_patt.loc6_48.2 (constants.%return.param_patt.2f0)]
// CHECK:STDOUT:     %return.patt.loc6_36.1: @F.%pattern_type.loc6_48 (%pattern_type.b5e) = return_slot_pattern %return.param_patt.loc6_48.1, %ptr.loc6_48.2 [symbolic = %return.patt.loc6_36.2 (constants.%return.patt.8e6)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_47: %Extendable.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T.44d)]
// CHECK:STDOUT:     %T.as_type.loc6_39: type = facet_access_type %T.ref.loc6_47 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)]
// CHECK:STDOUT:     %.loc6_39.2: type = converted %T.ref.loc6_47, %T.as_type.loc6_39 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)]
// CHECK:STDOUT:     %.loc6_39.3: type = partial_type %.loc6_39.2 [symbolic = %.loc6_39.1 (constants.%.437)]
// CHECK:STDOUT:     %ptr.loc6_48.2: type = ptr_type %.loc6_39.3 [symbolic = %ptr.loc6_48.1 (constants.%ptr.154)]
// CHECK:STDOUT:     %.loc6_48.2: Core.Form = init_form %ptr.loc6_48.2 [symbolic = %.loc6_48.1 (constants.%.970)]
// CHECK:STDOUT:     %.loc6_13: type = splice_block %Extendable.ref [concrete = constants.%Extendable.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Extendable.ref: type = name_ref Extendable, imports.%Core.Extendable [concrete = constants.%Extendable.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: %Extendable.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T.44d)]
// CHECK:STDOUT:     %t.param: ref @F.%T.as_type.loc6_33.1 (%T.as_type.43e) = ref_param call_param0
// CHECK:STDOUT:     %.loc6_33.1: type = splice_block %.loc6_33.2 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)] {
// CHECK:STDOUT:       %T.ref.loc6_33: %Extendable.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T.44d)]
// CHECK:STDOUT:       %T.as_type.loc6_33.2: type = facet_access_type %T.ref.loc6_33 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)]
// CHECK:STDOUT:       %.loc6_33.2: type = converted %T.ref.loc6_33, %T.as_type.loc6_33.2 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: ref @F.%T.as_type.loc6_33.1 (%T.as_type.43e) = wrapper_binding t, %t.param
// CHECK:STDOUT:     %return.param: ref @F.%ptr.loc6_48.1 (%ptr.154) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%ptr.loc6_48.1 (%ptr.154) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: %Extendable.type) {
// CHECK:STDOUT:   %T.patt.loc6_7.2: %pattern_type.7e1 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc6_7.2 (constants.%T.patt.993)]
// CHECK:STDOUT:   %T.loc6_7.1: %Extendable.type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T.44d)]
// CHECK:STDOUT:   %T.as_type.loc6_33.1: type = facet_access_type %T.loc6_7.1 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)]
// CHECK:STDOUT:   %pattern_type.loc6_31: type = pattern_type %T.as_type.loc6_33.1 [symbolic = %pattern_type.loc6_31 (constants.%pattern_type.7a5)]
// CHECK:STDOUT:   %t.param_patt.loc6_31.2: @F.%pattern_type.loc6_31 (%pattern_type.7a5) = ref_param_pattern [symbolic = %t.param_patt.loc6_31.2 (constants.%t.param_patt.0cf)]
// CHECK:STDOUT:   %t.patt.loc6_31.2: @F.%pattern_type.loc6_31 (%pattern_type.7a5) = wrapper_binding_pattern t, %t.param_patt.loc6_31.2 [symbolic = %t.patt.loc6_31.2 (constants.%t.patt.cb0)]
// CHECK:STDOUT:   %.loc6_39.1: type = partial_type %T.as_type.loc6_33.1 [symbolic = %.loc6_39.1 (constants.%.437)]
// CHECK:STDOUT:   %ptr.loc6_48.1: type = ptr_type %.loc6_39.1 [symbolic = %ptr.loc6_48.1 (constants.%ptr.154)]
// CHECK:STDOUT:   %.loc6_48.1: Core.Form = init_form %ptr.loc6_48.1 [symbolic = %.loc6_48.1 (constants.%.970)]
// CHECK:STDOUT:   %pattern_type.loc6_48: type = pattern_type %ptr.loc6_48.1 [symbolic = %pattern_type.loc6_48 (constants.%pattern_type.b5e)]
// CHECK:STDOUT:   %return.param_patt.loc6_48.2: @F.%pattern_type.loc6_48 (%pattern_type.b5e) = out_param_pattern [symbolic = %return.param_patt.loc6_48.2 (constants.%return.param_patt.2f0)]
// CHECK:STDOUT:   %return.patt.loc6_36.2: @F.%pattern_type.loc6_48 (%pattern_type.b5e) = return_slot_pattern %return.param_patt.loc6_48.2, %ptr.loc6_48.1 [symbolic = %return.patt.loc6_36.2 (constants.%return.patt.8e6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_31: <witness> = require_complete_type %T.as_type.loc6_33.1 [symbolic = %require_complete.loc6_31 (constants.%require_complete.b36)]
// CHECK:STDOUT:   %require_complete.loc6_36: <witness> = require_complete_type %ptr.loc6_48.1 [symbolic = %require_complete.loc6_36 (constants.%require_complete.f97)]
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %.loc6_39.1 [symbolic = %require_complete.loc7 (constants.%require_complete.ed9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc6_48.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.858)]
// CHECK:STDOUT:   %Copy.facet.loc7_10.3: %Copy.type = facet_value %ptr.loc6_48.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet.fb9)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc7_10.3) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.75d)]
// CHECK:STDOUT:   %.loc7_10.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc7_10.3 [symbolic = %.loc7_10.3 (constants.%.ae6)]
// CHECK:STDOUT:   %impl.elem0.loc7_10.2: @F.%.loc7_10.3 (%.ae6) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_10.2 (constants.%impl.elem0.f56)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_10.2: <specific function> = specific_impl_function %impl.elem0.loc7_10.2, @Copy.WithSelf.Op(%Copy.facet.loc7_10.3) [symbolic = %specific_impl_fn.loc7_10.2 (constants.%specific_impl_fn.e6e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%t.param: ref @F.%T.as_type.loc6_33.1 (%T.as_type.43e)) -> out %return.param: @F.%ptr.loc6_48.1 (%ptr.154) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %t.ref: ref @F.%T.as_type.loc6_33.1 (%T.as_type.43e) = name_ref t, %t
// CHECK:STDOUT:     %T.ref.loc7: %Extendable.type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T.44d)]
// CHECK:STDOUT:     %T.as_type.loc7: type = facet_access_type %T.ref.loc7 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)]
// CHECK:STDOUT:     %.loc7_17.1: type = converted %T.ref.loc7, %T.as_type.loc7 [symbolic = %T.as_type.loc6_33.1 (constants.%T.as_type.43e)]
// CHECK:STDOUT:     %.loc7_17.2: type = partial_type %.loc7_17.1 [symbolic = %.loc6_39.1 (constants.%.437)]
// CHECK:STDOUT:     %.loc7_14.1: ref @F.%.loc6_39.1 (%.437) = as_compatible %t.ref
// CHECK:STDOUT:     %.loc7_14.2: ref @F.%.loc6_39.1 (%.437) = converted %t.ref, %.loc7_14.1
// CHECK:STDOUT:     %addr: @F.%ptr.loc6_48.1 (%ptr.154) = addr_of %.loc7_14.2
// CHECK:STDOUT:     %impl.elem0.loc7_10.1: @F.%.loc7_10.3 (%.ae6) = impl_witness_access constants.%Copy.lookup_impl_witness.858, element0 [symbolic = %impl.elem0.loc7_10.2 (constants.%impl.elem0.f56)]
// CHECK:STDOUT:     %bound_method.loc7_10.1: <bound method> = bound_method %addr, %impl.elem0.loc7_10.1
// CHECK:STDOUT:     %Copy.facet.loc7_10.1: %Copy.type = facet_value constants.%ptr.154, (constants.%Copy.lookup_impl_witness.858) [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet.fb9)]
// CHECK:STDOUT:     %.loc7_10.1: %Copy.type = converted constants.%ptr.154, %Copy.facet.loc7_10.1 [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet.fb9)]
// CHECK:STDOUT:     %Copy.facet.loc7_10.2: %Copy.type = facet_value constants.%ptr.154, (constants.%Copy.lookup_impl_witness.858) [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet.fb9)]
// CHECK:STDOUT:     %.loc7_10.2: %Copy.type = converted constants.%ptr.154, %Copy.facet.loc7_10.2 [symbolic = %Copy.facet.loc7_10.3 (constants.%Copy.facet.fb9)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_10.1: <specific function> = specific_impl_function %impl.elem0.loc7_10.1, @Copy.WithSelf.Op(constants.%Copy.facet.fb9) [symbolic = %specific_impl_fn.loc7_10.2 (constants.%specific_impl_fn.e6e)]
// CHECK:STDOUT:     %bound_method.loc7_10.2: <bound method> = bound_method %addr, %specific_impl_fn.loc7_10.1
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @F.%ptr.loc6_48.1 (%ptr.154) = call %bound_method.loc7_10.2(%addr)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%b.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %b.ref: ref %B = name_ref b, %b
// CHECK:STDOUT:   %.loc13_5: %B = ref_tag %b.ref
// CHECK:STDOUT:   %Extendable.facet.loc13_10.1: %Extendable.type = facet_value constants.%B, (constants.%custom_witness) [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %.loc13_10.1: %Extendable.type = converted constants.%B, %Extendable.facet.loc13_10.1 [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %Extendable.facet.loc13_10.2: %Extendable.type = facet_value constants.%B, (constants.%custom_witness) [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %.loc13_10.2: %Extendable.type = converted constants.%B, %Extendable.facet.loc13_10.2 [concrete = constants.%Extendable.facet]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%Extendable.facet) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %ptr.3dc = call %F.specific_fn(%b.ref)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.44d) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt.993
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T.44d
// CHECK:STDOUT:   %T.as_type.loc6_33.1 => constants.%T.as_type.43e
// CHECK:STDOUT:   %pattern_type.loc6_31 => constants.%pattern_type.7a5
// CHECK:STDOUT:   %t.param_patt.loc6_31.2 => constants.%t.param_patt.0cf
// CHECK:STDOUT:   %t.patt.loc6_31.2 => constants.%t.patt.cb0
// CHECK:STDOUT:   %.loc6_39.1 => constants.%.437
// CHECK:STDOUT:   %ptr.loc6_48.1 => constants.%ptr.154
// CHECK:STDOUT:   %.loc6_48.1 => constants.%.970
// CHECK:STDOUT:   %pattern_type.loc6_48 => constants.%pattern_type.b5e
// CHECK:STDOUT:   %return.param_patt.loc6_48.2 => constants.%return.param_patt.2f0
// CHECK:STDOUT:   %return.patt.loc6_36.2 => constants.%return.patt.8e6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Extendable.facet) {
// CHECK:STDOUT:   %T.patt.loc6_7.2 => constants.%T.patt.993
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%Extendable.facet
// CHECK:STDOUT:   %T.as_type.loc6_33.1 => constants.%B
// CHECK:STDOUT:   %pattern_type.loc6_31 => constants.%pattern_type.e39
// CHECK:STDOUT:   %t.param_patt.loc6_31.2 => constants.%t.param_patt.375
// CHECK:STDOUT:   %t.patt.loc6_31.2 => constants.%t.patt.cab
// CHECK:STDOUT:   %.loc6_39.1 => constants.%.640
// CHECK:STDOUT:   %ptr.loc6_48.1 => constants.%ptr.3dc
// CHECK:STDOUT:   %.loc6_48.1 => constants.%.3c5
// CHECK:STDOUT:   %pattern_type.loc6_48 => constants.%pattern_type.777
// CHECK:STDOUT:   %return.param_patt.loc6_48.2 => constants.%return.param_patt.e41
// CHECK:STDOUT:   %return.patt.loc6_36.2 => constants.%return.patt.7d3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_31 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc6_36 => constants.%complete_type.df9
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type.357
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.45a
// CHECK:STDOUT:   %Copy.facet.loc7_10.3 => constants.%Copy.facet.8c6
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.b75
// CHECK:STDOUT:   %.loc7_10.3 => constants.%.a42
// CHECK:STDOUT:   %impl.elem0.loc7_10.2 => constants.%ptr.as.Copy.impl.Op.652
// CHECK:STDOUT:   %specific_impl_fn.loc7_10.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:

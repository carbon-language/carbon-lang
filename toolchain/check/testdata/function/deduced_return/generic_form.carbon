// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/generic_form.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/generic_form.carbon

// --- ref_auto.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[T: type](ref x: T) -> ref auto { return x; }

fn G(ref n: i32) -> ref i32 { return F(ref n); }
//@dump-sem-ir-end

// --- ref_auto_call_from_generic.carbon
library "[[@TEST_NAME]]";

fn F[T: type](ref x: T) -> ref auto { return x; }

//@dump-sem-ir-begin
fn G[U: type](ref u: U) -> ref U { return F(ref u); }
//@dump-sem-ir-end

fn H(ref n: i32) -> ref i32 { return G(ref n); }

// --- ref_auto_address_of_result.carbon
library "[[@TEST_NAME]]";

fn F[T: type](ref x: T) -> ref auto { return x; }

// The call is a durable reference expression, so its address can be taken.
fn G(ref n: i32) -> i32* { return &F(ref n); }
fn H[U: type](ref u: U) -> U* { return &F(ref u); }

// --- ref_auto_deduced_from_call.carbon
library "[[@TEST_NAME]]";

fn F[T: type](ref x: T) -> ref auto { return x; }
fn G[U: type](ref u: U) -> ref auto { return F(ref u); }

fn H(ref n: i32) -> i32* { return &G(ref n); }
fn I[V: type](ref v: V) -> V* { return &G(ref v); }

// --- ref_auto_return_type_from_definition.carbon
library "[[@TEST_NAME]]";

class C(T: type) {
  var x: T;
}

// The return type doesn't appear in the declaration, so is only computed by the
// definition's eval block.
fn F[T: type](ref c: C(T)) -> ref auto { return c.x; }

fn G(ref c: C(i32)) -> i32* { return &F(ref c); }
fn H[U: type](ref c: C(U)) -> U* { return &F(ref c); }

// --- ref_auto_member_of_generic_class.carbon
library "[[@TEST_NAME]]";

class C(T: type) {
  fn Get(ref self) -> ref auto { return self.x; }
  var x: T;
}

fn G(ref c: C(i32)) -> i32* { return &c.Get(); }
fn H[U: type](ref c: C(U)) -> U* { return &c.Get(); }

// --- val_auto.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[T: type](x: T) -> val auto { return x; }

fn G() {
  let unused n: i32 = F(1 as i32);
}
//@dump-sem-ir-end

fn H[U: type](u: U) {
  let unused v: U = F(u);
}

// --- var_auto.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T) -> var auto { return x; }

fn G() {
  var unused n: i32 = F(1 as i32);
}

fn H[U: Core.Copy & Core.Destroy](u: U) {
  var unused v: U = F(u);
}

// --- var_auto_returned_var.carbon
library "[[@TEST_NAME]]";

fn F[T: Core.Copy](x: T) -> var auto {
  returned var v: T = x;
  return var;
}

fn G() -> i32 { return F(1 as i32); }
fn H[U: Core.Copy](u: U) -> U { return F(u); }

// --- template_ref_auto.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](ref x: T) -> ref auto { return x.n; }

class C {
  var n: i32;
}

fn G(ref c: C) -> i32* { return &F(ref c); }
fn H(ref s: {.n: bool}) -> bool* { return &F(ref s); }

// --- template_val_auto.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> val auto { return x.n; }

class C {
  var n: i32;
}

fn G(c: C) -> i32 { return F(c); }

// --- template_ref_auto_call_from_checked_generic.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](ref x: T) -> ref auto { return x; }

fn G[U: type](ref u: U) -> U* { return &F(ref u); }

fn H(ref n: i32) -> i32* { return G(ref n); }

// --- fail_ref_auto_non_ref.carbon
library "[[@TEST_NAME]]";

fn F[T: type](x: T) -> ref auto {
  // CHECK:STDERR:        ╭─┤ fail_ref_auto_non_ref.carbon:[[@LINE+9]]:3
  // CHECK:STDERR: ->     │   return x;
  // CHECK:STDERR:        ·   ━━━━┳━━┯━
  // CHECK:STDERR:        ·       ┗━━━━━┫ error: cannot bind durable reference to non-reference value of
  // CHECK:STDERR:        ·          │  ┃        type `T`
  // CHECK:STDERR:        ·          │  ┃        [ConversionFailureNonRefToRef]
  // CHECK:STDERR:        ·          ╰────┤ return type deduced as `T` [ReturnTypeDeducedAs]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return x;
}

// --- fail_template_ref_auto_non_ref.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> ref auto { return x.n; }

class C {
  var n: i32;
}

fn G(c: C) {
  // CHECK:STDERR:        ╭─┤ fail_template_ref_auto_non_ref.carbon:[[@LINE+13]]:3
  // CHECK:STDERR:        │ fn F[template T: type](x: T) -> ref auto { return x.n; }
  // CHECK:STDERR:        ·                                            ┬──────────
  // CHECK:STDERR:        ·                                            ╰──┤ cannot bind durable reference to
  // CHECK:STDERR:        ·                                               │ non-reference value of type
  // CHECK:STDERR:        ·                                               │ `i32`
  // CHECK:STDERR:        ·                                               │ [ConversionFailureNonRefToRef]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(c);
  // CHECK:STDERR:        ·   ━━┳━
  // CHECK:STDERR:        ·     ┗━━┫ error: unable to monomorphize specific `F(C)` [ResolvingSpecificHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(c);
}

// --- fail_template_ref_auto_recursion_before_deduced.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](ref x: T) -> ref auto {
  return F(ref x);
}

fn G(ref n: i32) {
  // CHECK:STDERR:        ╭─┤ fail_template_ref_auto_recursion_before_deduced.carbon:[[@LINE+16]]:3
  // CHECK:STDERR:        │ fn F[template T: type](ref x: T) -> ref auto {
  // CHECK:STDERR:        · ───────────────────────┬──────────────────────
  // CHECK:STDERR:        ·                        ╰──┤ return type of `F` is deduced from its definition
  // CHECK:STDERR:        ·                           │ [ReturnTypeDeducedFromDefinition]
  // CHECK:STDERR:        │   return F(ref x);
  // CHECK:STDERR:        ·          ────┬───
  // CHECK:STDERR:        ·              ╰──┤ call to `F` before its return type is known
  // CHECK:STDERR:        ·                 │ [CallBeforeReturnTypeDeduced]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(ref n);
  // CHECK:STDERR:        ·   ━━━━┳━━━
  // CHECK:STDERR:        ·       ┗━━┫ error: unable to monomorphize specific `F(i32)`
  // CHECK:STDERR:        ·          ┃        [ResolvingSpecificHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(ref n);
}

// --- lib.carbon
library "[[@TEST_NAME]]";

fn RefId[T: type](ref x: T) -> ref auto { return x; }
fn ValId[T: type](x: T) -> val auto { return x; }

// --- use_lib.carbon
library "[[@TEST_NAME]]";

import library "lib";

fn G(ref n: i32) -> i32* { return &RefId(ref n); }

fn H() {
  let unused v: i32 = ValId(1 as i32);
}

fn I[U: type](ref u: U) -> U* { return &RefId(ref u); }

// CHECK:STDOUT: --- ref_auto.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %x.param_patt.2ca: %pattern_type.51d = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.c65: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt.2ca [symbolic]
// CHECK:STDOUT:   %.010: Core.Form = ref_form auto [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %.b33: Core.Form = ref_form %T [symbolic]
// CHECK:STDOUT:   %.066: %pattern_type.51d = ref_return_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.1cc: %pattern_type.51d = return_slot_pattern %.066, %T [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.param_patt: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete]
// CHECK:STDOUT:   %.30e: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %.b71: %pattern_type.d94 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.2c5: %pattern_type.d94 = return_slot_pattern %.b71, %i32 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %x.param_patt.85b: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.fe0: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt.85b [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc4_20.1: @F.%pattern_type (%pattern_type.51d) = ref_param_pattern [symbolic = %x.param_patt.loc4_20.2 (constants.%x.param_patt.2ca)]
// CHECK:STDOUT:     %x.patt.loc4_20.1: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc4_20.1 [symbolic = %x.patt.loc4_20.2 (constants.%x.patt.c65)]
// CHECK:STDOUT:     %.loc4_46.1: @F.%pattern_type (%pattern_type.51d) = ref_return_pattern [symbolic = %.loc4_46.3 (constants.%.066)]
// CHECK:STDOUT:     %return.patt.loc4_46.1: @F.%pattern_type (%pattern_type.51d) = return_slot_pattern %.loc4_46.1, %.loc4_46.4 [symbolic = %return.patt.loc4_46.2 (constants.%return.patt.1cc)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_28: Core.Form = ref_form auto [concrete = constants.%.010]
// CHECK:STDOUT:     %.loc4_9.1: type = splice_block %.loc4_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: ref @F.%T.loc4_7.1 (%T) = ref_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %x: ref @F.%T.loc4_7.1 (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc4_46.4: type = type_literal constants.%T [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_46.5: Core.Form = ref_form %.loc4_46.4 [symbolic = %.loc4_46.2 (constants.%.b33)]
// CHECK:STDOUT:     %.loc4_46.6: ref @F.%T.loc4_7.1 (%T) = ref_return
// CHECK:STDOUT:     %return: ref @F.%T.loc4_7.1 (%T) = return_slot %.loc4_46.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %n.param_patt: %pattern_type.d94 = ref_param_pattern [concrete = constants.%n.param_patt]
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = wrapper_binding_pattern n, %n.param_patt [concrete = constants.%n.patt]
// CHECK:STDOUT:     %.loc6_21.1: %pattern_type.d94 = ref_return_pattern [concrete = constants.%.b71]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %.loc6_21.1, %.loc6_21.3 [concrete = constants.%return.patt.2c5]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_21.2: Core.Form = ref_form %i32.loc6_25 [concrete = constants.%.30e]
// CHECK:STDOUT:     %.loc6_21.3: type = type_component_of %.loc6_21.2 [concrete = constants.%i32]
// CHECK:STDOUT:     %n.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %i32.loc6_13: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: ref %i32 = wrapper_binding n, %n.param
// CHECK:STDOUT:     %.loc6_21.4: ref %i32 = ref_return
// CHECK:STDOUT:     %return: ref %i32 = return_slot %.loc6_21.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_7.1 [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc4_20.2: @F.%pattern_type (%pattern_type.51d) = ref_param_pattern [symbolic = %x.param_patt.loc4_20.2 (constants.%x.param_patt.2ca)]
// CHECK:STDOUT:   %x.patt.loc4_20.2: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc4_20.2 [symbolic = %x.patt.loc4_20.2 (constants.%x.patt.c65)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_7.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc4_46.2: Core.Form = ref_form %T.loc4_7.1 [symbolic = %.loc4_46.2 (constants.%.b33)]
// CHECK:STDOUT:   %.loc4_46.3: @F.%pattern_type (%pattern_type.51d) = ref_return_pattern [symbolic = %.loc4_46.3 (constants.%.066)]
// CHECK:STDOUT:   %return.patt.loc4_46.2: @F.%pattern_type (%pattern_type.51d) = return_slot_pattern %.loc4_46.3, %T.loc4_7.1 [symbolic = %return.patt.loc4_46.2 (constants.%return.patt.1cc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: ref @F.%T.loc4_7.1 (%T)) -> ref %T [deduced] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: ref @F.%T.loc4_7.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc4_47: ref @F.%T.loc4_7.1 (%T) = splice_block %x.ref {}
// CHECK:STDOUT:     return %.loc4_47
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%n.param: ref %i32) -> ref %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %n.ref: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc6_40: %i32 = ref_tag %n.ref
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: ref %i32 = call %F.specific_fn(%n.ref)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc4_20.2 => constants.%x.param_patt.2ca
// CHECK:STDOUT:   %x.patt.loc4_20.2 => constants.%x.patt.c65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %x.param_patt.loc4_20.2 => constants.%x.param_patt.85b
// CHECK:STDOUT:   %x.patt.loc4_20.2 => constants.%x.patt.fe0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc4_46.2 => constants.%.30e
// CHECK:STDOUT:   %.loc4_46.3 => constants.%.b71
// CHECK:STDOUT:   %return.patt.loc4_46.2 => constants.%return.patt.2c5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- ref_auto_call_from_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %u.param_patt.5e6: %pattern_type.51d1c4.2 = ref_param_pattern [symbolic]
// CHECK:STDOUT:   %u.patt.c4b: %pattern_type.51d1c4.2 = wrapper_binding_pattern u, %u.param_patt.5e6 [symbolic]
// CHECK:STDOUT:   %.b33a4e.2: Core.Form = ref_form %U [symbolic]
// CHECK:STDOUT:   %.06682e.2: %pattern_type.51d1c4.2 = ref_return_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.1cce12.2: %pattern_type.51d1c4.2 = return_slot_pattern %.06682e.2, %U [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9443dd.2: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %F.specific_fn.c05: <specific function> = specific_function %F, @F(%U) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.30e: Core.Form = ref_form %i32 [concrete]
// CHECK:STDOUT:   %.b71: %pattern_type.d94 = ref_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.2c5: %pattern_type.d94 = return_slot_pattern %.b71, %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %u.param_patt.508: %pattern_type.d94 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %u.patt.4fa: %pattern_type.d94 = wrapper_binding_pattern u, %u.param_patt.508 [concrete]
// CHECK:STDOUT:   %F.specific_fn.9db: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %U.patt.loc6_7.1: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_7.2 (constants.%U.patt)]
// CHECK:STDOUT:     %u.param_patt.loc6_20.1: @G.%pattern_type (%pattern_type.51d1c4.2) = ref_param_pattern [symbolic = %u.param_patt.loc6_20.2 (constants.%u.param_patt.5e6)]
// CHECK:STDOUT:     %u.patt.loc6_20.1: @G.%pattern_type (%pattern_type.51d1c4.2) = wrapper_binding_pattern u, %u.param_patt.loc6_20.1 [symbolic = %u.patt.loc6_20.2 (constants.%u.patt.c4b)]
// CHECK:STDOUT:     %.loc6_28.1: @G.%pattern_type (%pattern_type.51d1c4.2) = ref_return_pattern [symbolic = %.loc6_28.3 (constants.%.06682e.2)]
// CHECK:STDOUT:     %return.patt.loc6_25.1: @G.%pattern_type (%pattern_type.51d1c4.2) = return_slot_pattern %.loc6_28.1, %.loc6_28.5 [symbolic = %return.patt.loc6_25.2 (constants.%return.patt.1cce12.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.ref.loc6_32: type = name_ref U, %U.loc6_7.2 [symbolic = %U.loc6_7.1 (constants.%U)]
// CHECK:STDOUT:     %.loc6_28.4: Core.Form = ref_form %U.ref.loc6_32 [symbolic = %.loc6_28.2 (constants.%.b33a4e.2)]
// CHECK:STDOUT:     %.loc6_28.5: type = type_component_of %.loc6_28.4 [symbolic = %U.loc6_7.1 (constants.%U)]
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc6_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc6_7.2: type = symbolic_binding U, 0 [symbolic = %U.loc6_7.1 (constants.%U)]
// CHECK:STDOUT:     %u.param: ref @G.%U.loc6_7.1 (%U) = ref_param call_param0
// CHECK:STDOUT:     %U.ref.loc6_22: type = name_ref U, %U.loc6_7.2 [symbolic = %U.loc6_7.1 (constants.%U)]
// CHECK:STDOUT:     %u: ref @G.%U.loc6_7.1 (%U) = wrapper_binding u, %u.param
// CHECK:STDOUT:     %.loc6_28.6: ref @G.%U.loc6_7.1 (%U) = ref_return
// CHECK:STDOUT:     %return: ref @G.%U.loc6_7.1 (%U) = return_slot %.loc6_28.6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%U.loc6_7.2: type) {
// CHECK:STDOUT:   %U.patt.loc6_7.2: %pattern_type.9a5 = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc6_7.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc6_7.1: type = symbolic_binding U, 0 [symbolic = %U.loc6_7.1 (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U.loc6_7.1 [symbolic = %pattern_type (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:   %u.param_patt.loc6_20.2: @G.%pattern_type (%pattern_type.51d1c4.2) = ref_param_pattern [symbolic = %u.param_patt.loc6_20.2 (constants.%u.param_patt.5e6)]
// CHECK:STDOUT:   %u.patt.loc6_20.2: @G.%pattern_type (%pattern_type.51d1c4.2) = wrapper_binding_pattern u, %u.param_patt.loc6_20.2 [symbolic = %u.patt.loc6_20.2 (constants.%u.patt.c4b)]
// CHECK:STDOUT:   %.loc6_28.2: Core.Form = ref_form %U.loc6_7.1 [symbolic = %.loc6_28.2 (constants.%.b33a4e.2)]
// CHECK:STDOUT:   %.loc6_28.3: @G.%pattern_type (%pattern_type.51d1c4.2) = ref_return_pattern [symbolic = %.loc6_28.3 (constants.%.06682e.2)]
// CHECK:STDOUT:   %return.patt.loc6_25.2: @G.%pattern_type (%pattern_type.51d1c4.2) = return_slot_pattern %.loc6_28.3, %U.loc6_7.1 [symbolic = %return.patt.loc6_25.2 (constants.%return.patt.1cce12.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U.loc6_7.1 [symbolic = %require_complete (constants.%require_complete.9443dd.2)]
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2: <specific function> = specific_function constants.%F, @F(%U.loc6_7.1) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%u.param: ref @G.%U.loc6_7.1 (%U)) -> ref %U {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %u.ref: ref @G.%U.loc6_7.1 (%U) = name_ref u, %u
// CHECK:STDOUT:     %.loc6_45: @G.%U.loc6_7.1 (%U) = ref_tag %u.ref
// CHECK:STDOUT:     %F.specific_fn.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%U) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:     %F.call: ref @G.%U.loc6_7.1 (%U) = call %F.specific_fn.loc6_43.1(%u.ref)
// CHECK:STDOUT:     return %F.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U.patt.loc6_7.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc6_7.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT:   %u.param_patt.loc6_20.2 => constants.%u.param_patt.5e6
// CHECK:STDOUT:   %u.patt.loc6_20.2 => constants.%u.patt.c4b
// CHECK:STDOUT:   %.loc6_28.2 => constants.%.b33a4e.2
// CHECK:STDOUT:   %.loc6_28.3 => constants.%.06682e.2
// CHECK:STDOUT:   %return.patt.loc6_25.2 => constants.%return.patt.1cce12.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %U.patt.loc6_7.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc6_7.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %u.param_patt.loc6_20.2 => constants.%u.param_patt.508
// CHECK:STDOUT:   %u.patt.loc6_20.2 => constants.%u.patt.4fa
// CHECK:STDOUT:   %.loc6_28.2 => constants.%.30e
// CHECK:STDOUT:   %.loc6_28.3 => constants.%.b71
// CHECK:STDOUT:   %return.patt.loc6_25.2 => constants.%return.patt.2c5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn.9db
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- val_auto.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %x.param_patt.91de5a.1: %pattern_type.51d1c4.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.26065e.1: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91de5a.1 [symbolic]
// CHECK:STDOUT:   %.0c3: Core.Form = value_form auto [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9443dd.1: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %.a89f79.1: Core.Form = value_form %T [symbolic]
// CHECK:STDOUT:   %.c5a291.1: %pattern_type.51d1c4.1 = value_return_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.5d1d5c.1: %pattern_type.51d1c4.1 = return_slot_pattern %.c5a291.1, %T [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %n.patt: %pattern_type.d94 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.b93: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.396: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.84a: %Core.IntLiteral.as.As.impl.Convert.type.396 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.impl_witness.e09: <witness> = impl_witness imports.%As.impl_witness_table.1c2, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type.2b0: type = fn_type @Core.IntLiteral.as.As.impl.Convert, @Core.IntLiteral.as.As.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.94f: %Core.IntLiteral.as.As.impl.Convert.type.2b0 = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.b93 = facet_value Core.IntLiteral, (%As.impl_witness.e09) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.2be: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %.979: type = fn_type_with_self_type %As.WithSelf.Convert.type.2be, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.94f [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.As.impl.Convert.94f, @Core.IntLiteral.as.As.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.68a: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %x.param_patt.051: %pattern_type.d94 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0f7: %pattern_type.d94 = wrapper_binding_pattern x, %x.param_patt.051 [concrete]
// CHECK:STDOUT:   %F.specific_fn.9db: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT:   %.acd: Core.Form = value_form %i32 [concrete]
// CHECK:STDOUT:   %.45b: %pattern_type.d94 = value_return_pattern [concrete]
// CHECK:STDOUT:   %return.patt.f8b: %pattern_type.d94 = return_slot_pattern %.45b, %i32 [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.3: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.9443dd.2: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %x.param_patt.91de5a.2: %pattern_type.51d1c4.3 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.26065e.2: %pattern_type.51d1c4.3 = wrapper_binding_pattern x, %x.param_patt.91de5a.2 [symbolic]
// CHECK:STDOUT:   %.a89f79.2: Core.Form = value_form %U [symbolic]
// CHECK:STDOUT:   %.c5a291.2: %pattern_type.51d1c4.3 = value_return_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.5d1d5c.2: %pattern_type.51d1c4.3 = return_slot_pattern %.c5a291.2, %U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.9e0: @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert.type (%Core.IntLiteral.as.As.impl.Convert.type.396) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.As.impl.%Core.IntLiteral.as.As.impl.Convert (constants.%Core.IntLiteral.as.As.impl.Convert.84a)]
// CHECK:STDOUT:   %As.impl_witness_table.1c2 = impl_witness_table (%Core.import_ref.9e0), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc4_16.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [symbolic = %x.param_patt.loc4_16.2 (constants.%x.param_patt.91de5a.1)]
// CHECK:STDOUT:     %x.patt.loc4_16.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc4_16.1 [symbolic = %x.patt.loc4_16.2 (constants.%x.patt.26065e.1)]
// CHECK:STDOUT:     %.loc4_42.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_return_pattern [symbolic = %.loc4_42.3 (constants.%.c5a291.1)]
// CHECK:STDOUT:     %return.patt.loc4_42.1: @F.%pattern_type (%pattern_type.51d1c4.1) = return_slot_pattern %.loc4_42.1, %.loc4_42.4 [symbolic = %return.patt.loc4_42.2 (constants.%return.patt.5d1d5c.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_24: Core.Form = value_form auto [concrete = constants.%.0c3]
// CHECK:STDOUT:     %.loc4_9.1: type = splice_block %.loc4_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_7.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_7.2 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc4_7.1 (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc4_42.4: type = type_literal constants.%T [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc4_42.5: Core.Form = value_form %.loc4_42.4 [symbolic = %.loc4_42.2 (constants.%.a89f79.1)]
// CHECK:STDOUT:     %.loc4_42.6: @F.%T.loc4_7.1 (%T) = value_return
// CHECK:STDOUT:     %return: @F.%T.loc4_7.1 (%T) = return_slot %.loc4_42.6
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_7.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_7.1 [symbolic = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc4_16.2: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [symbolic = %x.param_patt.loc4_16.2 (constants.%x.param_patt.91de5a.1)]
// CHECK:STDOUT:   %x.patt.loc4_16.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc4_16.2 [symbolic = %x.patt.loc4_16.2 (constants.%x.patt.26065e.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_7.1 [symbolic = %require_complete (constants.%require_complete.9443dd.1)]
// CHECK:STDOUT:   %.loc4_42.2: Core.Form = value_form %T.loc4_7.1 [symbolic = %.loc4_42.2 (constants.%.a89f79.1)]
// CHECK:STDOUT:   %.loc4_42.3: @F.%pattern_type (%pattern_type.51d1c4.1) = value_return_pattern [symbolic = %.loc4_42.3 (constants.%.c5a291.1)]
// CHECK:STDOUT:   %return.patt.loc4_42.2: @F.%pattern_type (%pattern_type.51d1c4.1) = return_slot_pattern %.loc4_42.3, %T.loc4_7.1 [symbolic = %return.patt.loc4_42.2 (constants.%return.patt.5d1d5c.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc4_7.1 (%T)) -> val %T [deduced] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_7.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %.loc4_43: @F.%T.loc4_7.1 (%T) = splice_block %x.ref {}
// CHECK:STDOUT:     return %.loc4_43
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %i32.loc7_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %impl.elem0: %.979 = impl_witness_access constants.%As.impl_witness.e09, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.94f]
// CHECK:STDOUT:   %bound_method.loc7_27.1: <bound method> = bound_method %int_1, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.As.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_27.2: <bound method> = bound_method %int_1, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call: init %i32 = call %bound_method.loc7_27.2(%int_1) [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc7_27.1: %i32 = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %.loc7_27.2: %i32 = converted %int_1, %.loc7_27.1 [concrete = constants.%int_1.68a]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn.9db]
// CHECK:STDOUT:   %F.call: %i32 = call %F.specific_fn(%.loc7_27.2)
// CHECK:STDOUT:   %i32.loc7_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %n: %i32 = wrapper_binding n, %F.call
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.d94 = value_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc4_16.2 => constants.%x.param_patt.91de5a.1
// CHECK:STDOUT:   %x.patt.loc4_16.2 => constants.%x.patt.26065e.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d94
// CHECK:STDOUT:   %x.param_patt.loc4_16.2 => constants.%x.param_patt.051
// CHECK:STDOUT:   %x.patt.loc4_16.2 => constants.%x.patt.0f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc4_42.2 => constants.%.acd
// CHECK:STDOUT:   %.loc4_42.3 => constants.%.45b
// CHECK:STDOUT:   %return.patt.loc4_42.2 => constants.%return.patt.f8b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%U) {
// CHECK:STDOUT:   %T.patt.loc4_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_7.1 => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.3
// CHECK:STDOUT:   %x.param_patt.loc4_16.2 => constants.%x.param_patt.91de5a.2
// CHECK:STDOUT:   %x.patt.loc4_16.2 => constants.%x.patt.26065e.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.9443dd.2
// CHECK:STDOUT:   %.loc4_42.2 => constants.%.a89f79.2
// CHECK:STDOUT:   %.loc4_42.3 => constants.%.c5a291.2
// CHECK:STDOUT:   %return.patt.loc4_42.2 => constants.%return.patt.5d1d5c.2
// CHECK:STDOUT: }
// CHECK:STDOUT:

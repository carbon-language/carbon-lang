// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/deduced_return/template.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/deduced_return/template.carbon

// --- member_access.carbon
library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[template T: type](x: T) -> auto { return x.n; }
//@dump-sem-ir-end

class C {
  var n: i32;
}

fn G(c: C) -> i32 { return F(c); }
fn H(s: {.n: bool}) -> bool { return F(s); }

// --- call_from_checked_generic.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> auto { return x; }

fn G[U: Core.Copy](u: U) -> U { return F(u); }

fn H() -> i32 { return G(0 as i32); }

// --- fail_instantiation_error.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> auto { return x.n; }

class D {
  var m: i32;
}

// The definition of `F(D)` is needed to determine the return type, so it's
// instantiated at the call.
fn G(d: D) -> i32 {
  // CHECK:STDERR:        ╭─┤ fail_instantiation_error.carbon:[[@LINE+13]]:10
  // CHECK:STDERR:        │ fn F[template T: type](x: T) -> auto { return x.n; }
  // CHECK:STDERR:        ·                                               ┬──
  // CHECK:STDERR:        ·                                               ╰──┤ member name `n` not found in
  // CHECK:STDERR:        ·                                                  │ `D`
  // CHECK:STDERR:        ·                                                  │ [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   return F(d);
  // CHECK:STDERR:        ·          ━━┳━
  // CHECK:STDERR:        ·            ┗━━┫ error: unable to monomorphize specific `F(D)`
  // CHECK:STDERR:        ·               ┃        [ResolvingSpecificHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return F(d);
}

// --- fail_recursion_before_deduced.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> auto {
  return F(x);
}

fn G() {
  // CHECK:STDERR:        ╭─┤ fail_recursion_before_deduced.carbon:[[@LINE+16]]:3
  // CHECK:STDERR:        │ fn F[template T: type](x: T) -> auto {
  // CHECK:STDERR:        · ───────────────────┬──────────────────
  // CHECK:STDERR:        ·                    ╰──┤ return type of `F` is deduced from its definition
  // CHECK:STDERR:        ·                       │ [ReturnTypeDeducedFromDefinition]
  // CHECK:STDERR:        │   return F(x);
  // CHECK:STDERR:        ·          ──┬─
  // CHECK:STDERR:        ·            ╰──┤ call to `F` before its return type is known
  // CHECK:STDERR:        ·               │ [CallBeforeReturnTypeDeduced]
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->     │   F(0 as i32);
  // CHECK:STDERR:        ·   ━━━━━┳━━━━━
  // CHECK:STDERR:        ·        ┗━━┫ error: unable to monomorphize specific `F(i32)`
  // CHECK:STDERR:        ·           ┃        [ResolvingSpecificHere]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  F(0 as i32);
}

// --- recursion_after_returned_var.carbon
library "[[@TEST_NAME]]";

fn F[template T: type](x: T, b: bool) -> auto {
  returned var r: T = x;
  if (b) {
    F(x, false);
  }
  return var;
}

fn G() -> i32 { return F(0 as i32, true); }

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %.237: type = type_of_inst @F.%.loc4_48.3 [template]
// CHECK:STDOUT:   %.f96: %.237 = splice_inst @F.%.loc4_48.3 [template]
// CHECK:STDOUT:   %.6c4: Core.Form = init_form %.237 [template]
// CHECK:STDOUT:   %pattern_type.940: type = pattern_type %.237 [template]
// CHECK:STDOUT:   %return.param_patt.bcf: %pattern_type.940 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.690: %pattern_type.940 = return_slot_pattern %return.param_patt.bcf, %.237 [template]
// CHECK:STDOUT:   %require_complete.b43: <witness> = require_complete_type %.237 [template]
// CHECK:STDOUT:   %.458: %.237 = splice_inst @F.%.loc4_50.3 [template]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n.4d4: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.31e: <witness> = complete_type_witness %struct_type.n.4d4 [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ab8: %i32 = splice_block %.5e4 {
// CHECK:STDOUT:       %.340: %C = specific_inst @F.%x.ref, @F(%C)
// CHECK:STDOUT:       %n.ref: %C.elem = name_ref n, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:       %.a79: ref %i32 = class_element_access %.340, element0
// CHECK:STDOUT:       %.5e4: %i32 = acquire_value %.a79
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.95d: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.95d) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet.95d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c58: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:       %impl.elem0.5a8: %.2b9 = impl_witness_access %Copy.impl_witness.8af, element0 [concrete = %Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:       %bound_method.ab3: <bound method> = bound_method %.ab8, %impl.elem0.5a8
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.5a8, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.764: <bound method> = bound_method %.ab8, %specific_fn
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.764(%.ab8)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %struct_type.n.ff2: type = struct_type {.n: bool} [concrete]
// CHECK:STDOUT:   %pattern_type.80d: type = pattern_type %struct_type.n.ff2 [concrete]
// CHECK:STDOUT:   %.f34: Core.Form = init_form bool [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %return.param_patt.5a1: %pattern_type.831 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c0c: %pattern_type.831 = return_slot_pattern %return.param_patt.5a1, bool [concrete]
// CHECK:STDOUT:   %x.param_patt.6cb: %pattern_type.80d = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.a4f: %pattern_type.80d = wrapper_binding_pattern x, %x.param_patt.6cb [concrete]
// CHECK:STDOUT:   %complete_type.ed1: <witness> = complete_type_witness %struct_type.n.ff2 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c30: bool = splice_block %.000 {
// CHECK:STDOUT:       %.ed5: %struct_type.n.ff2 = specific_inst @F.%x.ref, @F(%struct_type.n.ff2)
// CHECK:STDOUT:       %.000: bool = struct_access %.ed5, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.127: <witness> = complete_type_witness bool [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.5e1: <witness> = impl_witness imports.%Copy.impl_witness_table.bda [concrete]
// CHECK:STDOUT:   %Copy.facet.7af: %Copy.type = facet_value bool, (%Copy.impl_witness.5e1) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c98: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.7af) [concrete]
// CHECK:STDOUT:   %.68a: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c98, %Copy.facet.7af [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op.type: type = fn_type @bool.as.Copy.impl.Op [concrete]
// CHECK:STDOUT:   %bool.as.Copy.impl.Op: %bool.as.Copy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d3f: init bool = splice_block %bool.as.Copy.impl.Op.call {
// CHECK:STDOUT:       %impl.elem0.6df: %.68a = impl_witness_access %Copy.impl_witness.5e1, element0 [concrete = %bool.as.Copy.impl.Op]
// CHECK:STDOUT:       %bound_method.450: <bound method> = bound_method %.c30, %impl.elem0.6df
// CHECK:STDOUT:       %bool.as.Copy.impl.Op.call: init bool = call %bound_method.450(%.c30)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.9c3: %bool.as.Copy.impl.Op.type = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [concrete = constants.%bool.as.Copy.impl.Op]
// CHECK:STDOUT:   %Copy.impl_witness_table.bda = impl_witness_table (%Core.import_ref.9c3), @bool.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc4_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc4_25.1: @F.%pattern_type.loc4_25 (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc4_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc4_25.1: @F.%pattern_type.loc4_25 (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc4_25.1 [template = %x.patt.loc4_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt.loc4_48.1: @F.%pattern_type.loc4_48 (%pattern_type.940) = out_param_pattern [template = %return.param_patt.loc4_48.2 (constants.%return.param_patt.bcf)]
// CHECK:STDOUT:     %return.patt.loc4_48.1: @F.%pattern_type.loc4_48 (%pattern_type.940) = return_slot_pattern %return.param_patt.loc4_48.1, %.loc4_48.7 [template = %return.patt.loc4_48.2 (constants.%return.patt.690)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_18.1: type = splice_block %.loc4_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_16.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @F.%T.loc4_16.1 (%T.67db0b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc4_48.7: type = type_literal constants.%.237 [template = %.loc4_48.4 (constants.%.237)]
// CHECK:STDOUT:     %.loc4_48.8: Core.Form = init_form %.loc4_48.7 [template = %.loc4_48.6 (constants.%.6c4)]
// CHECK:STDOUT:     %return.param: ref @F.%.loc4_48.4 (%.237) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%.loc4_48.4 (%.237) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc4_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type.loc4_25: type = pattern_type %T.loc4_16.1 [template = %pattern_type.loc4_25 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc4_25.2: @F.%pattern_type.loc4_25 (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc4_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc4_25.2: @F.%pattern_type.loc4_25 (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc4_25.2 [template = %x.patt.loc4_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_25: <witness> = require_complete_type %T.loc4_16.1 [template = %require_complete.loc4_25 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc4_48.3: <instruction> = access_member_action %x.ref, n [template]
// CHECK:STDOUT:   %.loc4_48.4: type = type_of_inst %.loc4_48.3 [template = %.loc4_48.4 (constants.%.237)]
// CHECK:STDOUT:   %.loc4_48.5: @F.%.loc4_48.4 (%.237) = splice_inst %.loc4_48.3 [template = %.loc4_48.5 (constants.%.f96)]
// CHECK:STDOUT:   %.loc4_48.6: Core.Form = init_form %.loc4_48.4 [template = %.loc4_48.6 (constants.%.6c4)]
// CHECK:STDOUT:   %pattern_type.loc4_48: type = pattern_type %.loc4_48.4 [template = %pattern_type.loc4_48 (constants.%pattern_type.940)]
// CHECK:STDOUT:   %return.param_patt.loc4_48.2: @F.%pattern_type.loc4_48 (%pattern_type.940) = out_param_pattern [template = %return.param_patt.loc4_48.2 (constants.%return.param_patt.bcf)]
// CHECK:STDOUT:   %return.patt.loc4_48.2: @F.%pattern_type.loc4_48 (%pattern_type.940) = return_slot_pattern %return.param_patt.loc4_48.2, %.loc4_48.4 [template = %return.patt.loc4_48.2 (constants.%return.patt.690)]
// CHECK:STDOUT:   %require_complete.loc4_48: <witness> = require_complete_type %.loc4_48.4 [template = %require_complete.loc4_48 (constants.%require_complete.b43)]
// CHECK:STDOUT:   %.loc4_50.3: <instruction> = convert_to_category_action %.loc4_48.2, element10 [template]
// CHECK:STDOUT:   %.loc4_50.4: @F.%.loc4_48.4 (%.237) = splice_inst %.loc4_50.3 [template = %.loc4_50.4 (constants.%.458)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc4_16.1 (%T.67db0b.1)) -> out %return.param: @F.%.loc4_48.4 (%.237) [deduced] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc4_48.1: type = type_of_inst %.loc4_48.3 [template = %.loc4_48.4 (constants.%.237)]
// CHECK:STDOUT:     %.loc4_48.2: @F.%.loc4_48.4 (%.237) = splice_inst %.loc4_48.3 [template = %.loc4_48.5 (constants.%.f96)]
// CHECK:STDOUT:     %.loc4_50.1: @F.%.loc4_48.4 (%.237) = splice_block %.loc4_50.2 [template = %.loc4_50.4 (constants.%.458)] {
// CHECK:STDOUT:       %.loc4_50.2: @F.%.loc4_48.4 (%.237) = splice_inst %.loc4_50.3 [template = %.loc4_50.4 (constants.%.458)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return %.loc4_50.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type.loc4_25 => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc4_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc4_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc4_25 => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc4_25.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc4_25.2 => constants.%x.patt.953
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_25 => constants.%complete_type.31e
// CHECK:STDOUT:   %.loc4_48.3 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc4_48.4 => constants.%i32
// CHECK:STDOUT:   %.loc4_48.5 => <not constant>
// CHECK:STDOUT:   %.loc4_48.6 => constants.%.0b3
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.d94
// CHECK:STDOUT:   %return.param_patt.loc4_48.2 => constants.%return.param_patt.299
// CHECK:STDOUT:   %return.patt.loc4_48.2 => constants.%return.patt.6fa
// CHECK:STDOUT:   %require_complete.loc4_48 => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc4_50.3 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc4_50.4 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct_type.n.ff2) {
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%struct_type.n.ff2
// CHECK:STDOUT:   %pattern_type.loc4_25 => constants.%pattern_type.80d
// CHECK:STDOUT:   %x.param_patt.loc4_25.2 => constants.%x.param_patt.6cb
// CHECK:STDOUT:   %x.patt.loc4_25.2 => constants.%x.patt.a4f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc4_25 => constants.%complete_type.ed1
// CHECK:STDOUT:   %.loc4_48.3 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc4_48.4 => bool
// CHECK:STDOUT:   %.loc4_48.5 => <not constant>
// CHECK:STDOUT:   %.loc4_48.6 => constants.%.f34
// CHECK:STDOUT:   %pattern_type.loc4_48 => constants.%pattern_type.831
// CHECK:STDOUT:   %return.param_patt.loc4_48.2 => constants.%return.param_patt.5a1
// CHECK:STDOUT:   %return.patt.loc4_48.2 => constants.%return.patt.c0c
// CHECK:STDOUT:   %require_complete.loc4_48 => constants.%complete_type.127
// CHECK:STDOUT:   %.loc4_50.3 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc4_50.4 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:

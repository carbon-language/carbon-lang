// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/call.carbon

// --- call.carbon

library "[[@TEST_NAME]]";

eval fn F() -> i32 { return 3; }

var a: array(i32, F()) = (0, 1, 2);

// --- fail_call_wrong_value.carbon

library "[[@TEST_NAME]]";

eval fn F() -> i32 { return 3; }

// Ensure we get an error about the over-sized initializer and aren't just
// treating all `eval fn` calls as silent errors.
// CHECK:STDERR: fail_call_wrong_value.carbon:[[@LINE+4]]:26: error: cannot initialize array of 3 elements from 4 initializers [ArrayInitFromLiteralArgCountMismatch]
// CHECK:STDERR: var a: array(i32, F()) = (0, 1, 2, 3);
// CHECK:STDERR:                          ^~~~~~~~~~~~
// CHECK:STDERR:
var a: array(i32, F()) = (0, 1, 2, 3);

// --- fail_call_fn_not_eval.carbon

library "[[@TEST_NAME]]";

fn F() -> i32 { return 3; }

// CHECK:STDERR: fail_call_fn_not_eval.carbon:[[@LINE+4]]:19: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var a: array(i32, F()) = (0, 1, 2);
// CHECK:STDERR:                   ^~~
// CHECK:STDERR:
var a: array(i32, F()) = (0, 1, 2);

// --- fail_call_eval_fn_not_defined.carbon

library "[[@TEST_NAME]]";

eval fn F() -> i32;

// TODO: We should be able to diagnose this better.
// CHECK:STDERR: fail_call_eval_fn_not_defined.carbon:[[@LINE+4]]:19: error: array bound is not a constant [InvalidArrayExpr]
// CHECK:STDERR: var a: array(i32, F()) = (0, 1, 2);
// CHECK:STDERR:                   ^~~
// CHECK:STDERR:
var a: array(i32, F()) = (0, 1, 2);

// --- param_and_return.carbon

library "[[@TEST_NAME]]";

eval fn F(x: i32) -> i32 { return x; }

var a: array(i32, F(3)) = (1, 2, 3);

// --- fail_todo_dependent_call.carbon

library "[[@TEST_NAME]]";

eval fn F(x: i32) -> i32 { return x; }

fn G(generic N: i32) {
  // CHECK:STDERR: fail_todo_dependent_call.carbon:[[@LINE+4]]:36: error: cannot initialize array with dependent bound from a list of initializers [ArrayInitDependentBound]
  // CHECK:STDERR:   var unused a: array(i32, F(N)) = (0, 1, 2);
  // CHECK:STDERR:                                    ^~~~~~~~~
  // CHECK:STDERR:
  var unused a: array(i32, F(N)) = (0, 1, 2);
}

fn H() { G(3); }

// --- dependent_call_type.carbon

library "[[@TEST_NAME]]";

class C {}
eval fn F(_: i32) -> type {
  return C;
}

//@dump-sem-ir-begin
fn UseFGenerically(generic X: i32) {
  var unused v: F(X) = {};
}
//@dump-sem-ir-end

fn UseFSpecifically() {
  UseFGenerically(3);
}

// --- return_in_place_discarded.carbon

library "[[@TEST_NAME]]";

eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }

fn G() {
  //@dump-sem-ir-begin
  F();
  //@dump-sem-ir-end
}

// --- return_in_place_init.carbon

library "[[@TEST_NAME]]";

eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }

fn G() {
  //@dump-sem-ir-begin
  var unused a: (i32, i32, i32) = F();
  //@dump-sem-ir-end
}

// --- fail_todo_return_in_place_tuple.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
eval fn F() -> (i32, i32, i32) { return (1, 2, 3); }
//@dump-sem-ir-end

eval fn G(v: (i32, i32, i32)) -> i32 { return v.1; }

fn H() {
  // TODO: The call to the function is a constant, but the `temporary` instruction wrapping it is not.
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_return_in_place_tuple.carbon:[[@LINE+4]]:28: error: array bound is not a constant [InvalidArrayExpr]
  // CHECK:STDERR:   var unused a: array(i32, G(F())) = (1, 2);
  // CHECK:STDERR:                            ^~~~~~
  // CHECK:STDERR:
  var unused a: array(i32, G(F())) = (1, 2);
  //@dump-sem-ir-end
}

// --- return_in_place_class.carbon

library "[[@TEST_NAME]]";

class C {}
//@dump-sem-ir-begin
eval fn F() -> C { return {}; }
//@dump-sem-ir-end

fn G(generic _: C) {}

fn H() {
  //@dump-sem-ir-begin
  G(F());
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- dependent_call_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %X.patt: %pattern_type.6b6 = symbolic_binding_pattern X, 0 [symbolic]
// CHECK:STDOUT:   %X: %i32 = symbolic_binding X, 0 [symbolic]
// CHECK:STDOUT:   %UseFGenerically.type: type = fn_type @UseFGenerically [concrete]
// CHECK:STDOUT:   %UseFGenerically: %UseFGenerically.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.call: init type = call %F(%X) [template]
// CHECK:STDOUT:   %require_complete.d08: <witness> = require_complete_type %F.call [template]
// CHECK:STDOUT:   %pattern_type.231: type = pattern_type %F.call [template]
// CHECK:STDOUT:   %v.patt.bcc: %pattern_type.231 = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt.53d: %pattern_type.231 = var_pattern %v.patt.bcc [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.4e499d.1: @UseFGenerically.%.loc11_3.12 (@UseFGenerically.%.loc11_3.12) = splice_inst @UseFGenerically.%.loc11_3.11 [template]
// CHECK:STDOUT:   %.6c2dfb.1: @UseFGenerically.%.loc11_3.15 (@UseFGenerically.%.loc11_3.15) = splice_inst @UseFGenerically.%.loc11_3.14 [template]
// CHECK:STDOUT:   %.031: @UseFGenerically.%.loc11_3.18 (@UseFGenerically.%.loc11_3.18) = splice_inst @UseFGenerically.%.loc11_3.17 [template]
// CHECK:STDOUT:   %.88d: @UseFGenerically.%.loc11_3.21 (@UseFGenerically.%.loc11_3.21) = splice_inst @UseFGenerically.%.loc11_3.20 [template]
// CHECK:STDOUT:   %.1d7: %F.call = splice_inst @UseFGenerically.%.loc11_3.23 [template]
// CHECK:STDOUT:   %.55e: %F.call = splice_inst @UseFGenerically.%.loc11_3.25 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.885: %F.call = splice_inst @UseFGenerically.%.loc11_3.27 [template]
// CHECK:STDOUT:   %.0c8: @UseFGenerically.%.loc11_3.30 (@UseFGenerically.%.loc11_3.30) = splice_inst @UseFGenerically.%.loc11_3.29 [template]
// CHECK:STDOUT:   %.6cb: @UseFGenerically.%.loc11_3.33 (@UseFGenerically.%.loc11_3.33) = splice_inst @UseFGenerically.%.loc11_3.32 [template]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %v.patt.5c6: %pattern_type.98b = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.a6f: %pattern_type.98b = var_pattern %v.patt.5c6 [concrete]
// CHECK:STDOUT:   %.2e6: <instruction> = call_action (%ImplicitAs.generic, %F.call), false [template]
// CHECK:STDOUT:   %.53a: type = type_of_inst %.2e6 [template]
// CHECK:STDOUT:   %.4e499d.2: %.53a = splice_inst %.2e6 [template]
// CHECK:STDOUT:   %.b1d: <instruction> = access_member_action %.4e499d.2, Convert [template]
// CHECK:STDOUT:   %.7df: type = type_of_inst %.b1d [template]
// CHECK:STDOUT:   %.6c2dfb.2: %.7df = splice_inst %.b1d [template]
// CHECK:STDOUT:   %.f27: <instruction> = compound_member_access_action %empty_struct, %.6c2dfb.2 [template]
// CHECK:STDOUT:   %.396: type = type_of_inst %.f27 [template]
// CHECK:STDOUT:   %.114: %.396 = splice_inst %.f27 [template]
// CHECK:STDOUT:   %.f4b: <instruction> = call_action (%.114), true [template]
// CHECK:STDOUT:   %.eb2: type = type_of_inst %.f4b [template]
// CHECK:STDOUT:   %.1c7: %.eb2 = splice_inst %.f4b [template]
// CHECK:STDOUT:   %.0b5: %C = splice_inst %.f4b [template]
// CHECK:STDOUT:   %inst.as_compatible.e86: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.daf: %C = as_compatible %.1c7 [template = %.0b5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.ecf: <instruction> = convert_to_category_action %.0b5, element10 [template]
// CHECK:STDOUT:   %.b86: %C = splice_inst %.ecf [template]
// CHECK:STDOUT:   %inst.as_compatible.9d0: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.9fb: ref %C = as_compatible @UseFGenerically.%v.var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc11_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.1a5: %Destroy.type = facet_value %C, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.7f2: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.1a5) [concrete]
// CHECK:STDOUT:   %.da8: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.7f2, %Destroy.facet.1a5 [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7ba: <bound method> = splice_block %bound_method.767 {
// CHECK:STDOUT:       %impl.elem0.96c: %.da8 = impl_witness_access %custom_witness.df9cc1.2, element0 [concrete = %Destroy.Op.1a2547.2]
// CHECK:STDOUT:       %bound_method.767: <bound method> = bound_method %.9fb, %impl.elem0.96c
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.Op.call: init %empty_tuple.type = call %.7ba(%.9fb)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %UseFGenerically.decl: %UseFGenerically.type = fn_decl @UseFGenerically [concrete = constants.%UseFGenerically] {
// CHECK:STDOUT:     %X.patt.loc10_29.1: %pattern_type.6b6 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc10_29.2 (constants.%X.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %X.loc10_29.2: %i32 = symbolic_binding X, 0 [symbolic = %X.loc10_29.1 (constants.%X)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @UseFGenerically(%X.loc10_29.2: %i32) {
// CHECK:STDOUT:   %X.patt.loc10_29.2: %pattern_type.6b6 = symbolic_binding_pattern X, 0 [symbolic = %X.patt.loc10_29.2 (constants.%X.patt)]
// CHECK:STDOUT:   %X.loc10_29.1: %i32 = symbolic_binding X, 0 [symbolic = %X.loc10_29.1 (constants.%X)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.call.loc11_20.2: init type = call constants.%F(%X.loc10_29.1) [template = %F.call.loc11_20.2 (constants.%F.call)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %F.call.loc11_20.2 [template = %require_complete (constants.%require_complete.d08)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %F.call.loc11_20.2 [template = %pattern_type (constants.%pattern_type.231)]
// CHECK:STDOUT:   %v.patt.loc11_15.2: @UseFGenerically.%pattern_type (%pattern_type.231) = ref_binding_pattern v [template = %v.patt.loc11_15.2 (constants.%v.patt.bcc)]
// CHECK:STDOUT:   %v.var_patt.loc11_3.2: @UseFGenerically.%pattern_type (%pattern_type.231) = var_pattern %v.patt.loc11_15.2 [template = %v.var_patt.loc11_3.2 (constants.%v.var_patt.53d)]
// CHECK:STDOUT:   %.loc11_3.11: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%F.call), false [template]
// CHECK:STDOUT:   %.loc11_3.12: type = type_of_inst %.loc11_3.11 [template]
// CHECK:STDOUT:   %.loc11_3.13: @UseFGenerically.%.loc11_3.12 (@UseFGenerically.%.loc11_3.12) = splice_inst %.loc11_3.11 [template = %.loc11_3.13 (constants.%.4e499d.1)]
// CHECK:STDOUT:   %.loc11_3.14: <instruction> = access_member_action %.loc11_3.1, Convert [template]
// CHECK:STDOUT:   %.loc11_3.15: type = type_of_inst %.loc11_3.14 [template]
// CHECK:STDOUT:   %.loc11_3.16: @UseFGenerically.%.loc11_3.15 (@UseFGenerically.%.loc11_3.15) = splice_inst %.loc11_3.14 [template = %.loc11_3.16 (constants.%.6c2dfb.1)]
// CHECK:STDOUT:   %.loc11_3.17: <instruction> = compound_member_access_action %.loc11_25, %.loc11_3.2 [template]
// CHECK:STDOUT:   %.loc11_3.18: type = type_of_inst %.loc11_3.17 [template]
// CHECK:STDOUT:   %.loc11_3.19: @UseFGenerically.%.loc11_3.18 (@UseFGenerically.%.loc11_3.18) = splice_inst %.loc11_3.17 [template = %.loc11_3.19 (constants.%.031)]
// CHECK:STDOUT:   %.loc11_3.20: <instruction> = call_action (%.loc11_3.3), true [template]
// CHECK:STDOUT:   %.loc11_3.21: type = type_of_inst %.loc11_3.20 [template]
// CHECK:STDOUT:   %.loc11_3.22: @UseFGenerically.%.loc11_3.21 (@UseFGenerically.%.loc11_3.21) = splice_inst %.loc11_3.20 [template = %.loc11_3.22 (constants.%.88d)]
// CHECK:STDOUT:   %.loc11_3.23: <instruction> = refine_type_action %.loc11_3.5, %F.call.loc11_20.2 [template]
// CHECK:STDOUT:   %.loc11_3.24: @UseFGenerically.%F.call.loc11_20.2 (%F.call) = splice_inst %.loc11_3.23 [template = %.loc11_3.24 (constants.%.1d7)]
// CHECK:STDOUT:   %.loc11_3.25: <instruction> = convert_to_category_action %.loc11_3.6, element10 [template]
// CHECK:STDOUT:   %.loc11_3.26: @UseFGenerically.%F.call.loc11_20.2 (%F.call) = splice_inst %.loc11_3.25 [template = %.loc11_3.26 (constants.%.55e)]
// CHECK:STDOUT:   %.loc11_3.27: <instruction> = refine_type_action %v.var, %F.call.loc11_20.2 [template]
// CHECK:STDOUT:   %.loc11_3.28: @UseFGenerically.%F.call.loc11_20.2 (%F.call) = splice_inst %.loc11_3.27 [template = %.loc11_3.28 (constants.%.885)]
// CHECK:STDOUT:   %.loc11_3.29: <instruction> = compound_member_access_action %.loc11_3.8, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc11_3.30: type = type_of_inst %.loc11_3.29 [template]
// CHECK:STDOUT:   %.loc11_3.31: @UseFGenerically.%.loc11_3.30 (@UseFGenerically.%.loc11_3.30) = splice_inst %.loc11_3.29 [template = %.loc11_3.31 (constants.%.0c8)]
// CHECK:STDOUT:   %.loc11_3.32: <instruction> = call_action (%.loc11_3.9), true [template]
// CHECK:STDOUT:   %.loc11_3.33: type = type_of_inst %.loc11_3.32 [template]
// CHECK:STDOUT:   %.loc11_3.34: @UseFGenerically.%.loc11_3.33 (@UseFGenerically.%.loc11_3.33) = splice_inst %.loc11_3.32 [template = %.loc11_3.34 (constants.%.6cb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @UseFGenerically.%F.call.loc11_20.2 (%F.call) = var_storage %v.var_patt.loc11_3.1
// CHECK:STDOUT:     %.loc11_25: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc11_3.1: @UseFGenerically.%.loc11_3.12 (@UseFGenerically.%.loc11_3.12) = splice_inst %.loc11_3.11 [template = %.loc11_3.13 (constants.%.4e499d.1)]
// CHECK:STDOUT:     %.loc11_3.2: @UseFGenerically.%.loc11_3.15 (@UseFGenerically.%.loc11_3.15) = splice_inst %.loc11_3.14 [template = %.loc11_3.16 (constants.%.6c2dfb.1)]
// CHECK:STDOUT:     %.loc11_3.3: @UseFGenerically.%.loc11_3.18 (@UseFGenerically.%.loc11_3.18) = splice_inst %.loc11_3.17 [template = %.loc11_3.19 (constants.%.031)]
// CHECK:STDOUT:     %.loc11_3.4: @UseFGenerically.%.loc11_3.21 (@UseFGenerically.%.loc11_3.21) = splice_inst %.loc11_3.20 [template = %.loc11_3.22 (constants.%.88d)]
// CHECK:STDOUT:     %.loc11_3.5: @UseFGenerically.%F.call.loc11_20.2 (%F.call) = converted %.loc11_25, %.loc11_3.4 [template = %.loc11_3.22 (constants.%.88d)]
// CHECK:STDOUT:     %.loc11_3.6: @UseFGenerically.%F.call.loc11_20.2 (%F.call) = splice_inst %.loc11_3.23 [template = %.loc11_3.24 (constants.%.1d7)]
// CHECK:STDOUT:     %.loc11_3.7: @UseFGenerically.%F.call.loc11_20.2 (%F.call) = splice_inst %.loc11_3.25 [template = %.loc11_3.26 (constants.%.55e)]
// CHECK:STDOUT:     assign %v.var, %.loc11_3.7
// CHECK:STDOUT:     %.loc11_20.1: type = splice_block %.loc11_20.3 [template = %F.call.loc11_20.2 (constants.%F.call)] {
// CHECK:STDOUT:       %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:       %X.ref: %i32 = name_ref X, %X.loc10_29.2 [symbolic = %X.loc10_29.1 (constants.%X)]
// CHECK:STDOUT:       %F.call.loc11_20.1: init type = call %F.ref(%X.ref) [template = %F.call.loc11_20.2 (constants.%F.call)]
// CHECK:STDOUT:       %.loc11_20.2: type = value_of_initializer %F.call.loc11_20.1 [template = %F.call.loc11_20.2 (constants.%F.call)]
// CHECK:STDOUT:       %.loc11_20.3: type = converted %F.call.loc11_20.1, %.loc11_20.2 [template = %F.call.loc11_20.2 (constants.%F.call)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @UseFGenerically.%F.call.loc11_20.2 (%F.call) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc11_15.1: @UseFGenerically.%pattern_type (%pattern_type.231) = ref_binding_pattern v [template = %v.patt.loc11_15.2 (constants.%v.patt.bcc)]
// CHECK:STDOUT:       %v.var_patt.loc11_3.1: @UseFGenerically.%pattern_type (%pattern_type.231) = var_pattern %v.patt.loc11_15.1 [template = %v.var_patt.loc11_3.2 (constants.%v.var_patt.53d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc11_3.8: @UseFGenerically.%F.call.loc11_20.2 (%F.call) = splice_inst %.loc11_3.27 [template = %.loc11_3.28 (constants.%.885)]
// CHECK:STDOUT:     %.loc11_3.9: @UseFGenerically.%.loc11_3.30 (@UseFGenerically.%.loc11_3.30) = splice_inst %.loc11_3.29 [template = %.loc11_3.31 (constants.%.0c8)]
// CHECK:STDOUT:     %.loc11_3.10: @UseFGenerically.%.loc11_3.33 (@UseFGenerically.%.loc11_3.33) = splice_inst %.loc11_3.32 [template = %.loc11_3.34 (constants.%.6cb)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc11_3.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseFGenerically(constants.%X) {
// CHECK:STDOUT:   %X.patt.loc10_29.2 => constants.%X.patt
// CHECK:STDOUT:   %X.loc10_29.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @UseFGenerically(constants.%int_3.410) {
// CHECK:STDOUT:   %X.patt.loc10_29.2 => constants.%X.patt
// CHECK:STDOUT:   %X.loc10_29.1 => constants.%int_3.410
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.call.loc11_20.2 => constants.%C
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %v.patt.loc11_15.2 => constants.%v.patt.5c6
// CHECK:STDOUT:   %v.var_patt.loc11_3.2 => constants.%v.var_patt.a6f
// CHECK:STDOUT:   %.loc11_3.11 => constants.%.2e6
// CHECK:STDOUT:   %.loc11_3.12 => constants.%.53a
// CHECK:STDOUT:   %.loc11_3.13 => constants.%.4e499d.2
// CHECK:STDOUT:   %.loc11_3.14 => constants.%.b1d
// CHECK:STDOUT:   %.loc11_3.15 => constants.%.7df
// CHECK:STDOUT:   %.loc11_3.16 => constants.%.6c2dfb.2
// CHECK:STDOUT:   %.loc11_3.17 => constants.%.f27
// CHECK:STDOUT:   %.loc11_3.18 => constants.%.396
// CHECK:STDOUT:   %.loc11_3.19 => constants.%.114
// CHECK:STDOUT:   %.loc11_3.20 => constants.%.f4b
// CHECK:STDOUT:   %.loc11_3.21 => constants.%.eb2
// CHECK:STDOUT:   %.loc11_3.22 => constants.%.1c7
// CHECK:STDOUT:   %.loc11_3.23 => constants.%inst.as_compatible.e86
// CHECK:STDOUT:   %.loc11_3.24 => constants.%.0b5
// CHECK:STDOUT:   %.loc11_3.25 => constants.%.ecf
// CHECK:STDOUT:   %.loc11_3.26 => constants.%.b86
// CHECK:STDOUT:   %.loc11_3.27 => constants.%inst.as_compatible.9d0
// CHECK:STDOUT:   %.loc11_3.28 => invalid
// CHECK:STDOUT:   %.loc11_3.29 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc11_3.30 => <bound method>
// CHECK:STDOUT:   %.loc11_3.31 => invalid
// CHECK:STDOUT:   %.loc11_3.32 => constants.%inst.call
// CHECK:STDOUT:   %.loc11_3.33 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc11_3.34 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_in_place_discarded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.555: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.479: %tuple.type.555 = tuple_value (%int_1.0c6, %int_2.295, %int_3.410) [concrete]
// CHECK:STDOUT:   %.e1e: ref %tuple.type.555 = temporary invalid, %tuple.479 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_5.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.e1e, %Destroy.Op.1a2547.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8_5.1: ref %tuple.type.555 = temporary_storage
// CHECK:STDOUT:   %F.call: init %tuple.type.555 to %.loc8_5.1 = call %F.ref() [concrete = constants.%tuple.479]
// CHECK:STDOUT:   %.loc8_5.2: ref %tuple.type.555 = temporary %.loc8_5.1, %F.call [concrete = constants.%.e1e]
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.e1e)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_5.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_5.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_5.3(%self.param: ref %tuple.type.555) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_in_place_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.04d: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.555: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %pattern_type.777: type = pattern_type %tuple.type.555 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.479: %tuple.type.555 = tuple_value (%int_1.0c6, %int_2.295, %int_3.410) [concrete]
// CHECK:STDOUT:   %a.patt: %pattern_type.777 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:   %a.var_patt: %pattern_type.777 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref %tuple.type.555 = var_storage %a.var_patt
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc8_3: ref %tuple.type.555 = splice_block %a.var {}
// CHECK:STDOUT:   %F.call: init %tuple.type.555 to %.loc8_3 = call %F.ref() [concrete = constants.%tuple.479]
// CHECK:STDOUT:   assign %a.var, %F.call
// CHECK:STDOUT:   %.loc8_31.1: type = splice_block %.loc8_31.3 [concrete = constants.%tuple.type.555] {
// CHECK:STDOUT:     %i32.loc8_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc8_23: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc8_28: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8_31.2: %tuple.type.ff9 = tuple_literal (%i32.loc8_18, %i32.loc8_23, %i32.loc8_28) [concrete = constants.%tuple.04d]
// CHECK:STDOUT:     %.loc8_31.3: type = converted %.loc8_31.2, constants.%tuple.type.555 [concrete = constants.%tuple.type.555]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %tuple.type.555 = wrapper_binding a, %a.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.777 = ref_binding_pattern a [concrete = constants.%a.patt]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.777 = var_pattern %a.patt [concrete = constants.%a.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %a.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%a.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %tuple.type.555) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_return_in_place_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %tuple.type.ff9: type = tuple_type (type, type, type) [concrete]
// CHECK:STDOUT:   %tuple.04d: %tuple.type.ff9 = tuple_value (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %tuple.type.555: type = tuple_type (%i32, %i32, %i32) [concrete]
// CHECK:STDOUT:   %.718: Core.Form = init_form %tuple.type.555 [concrete]
// CHECK:STDOUT:   %pattern_type.777: type = pattern_type %tuple.type.555 [concrete]
// CHECK:STDOUT:   %return.param_patt.789: %pattern_type.777 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e16: %pattern_type.777 = return_slot_pattern %return.param_patt.789, %tuple.type.555 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.type.37f: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.2d5: %tuple.type.37f = tuple_value (%int_1.5b8, %int_2.ecc, %int_3.1ba) [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.953: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_1.0c6: %i32 = int_value 1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %tuple.479: %tuple.type.555 = tuple_value (%int_1.0c6, %int_2.295, %int_3.410) [concrete]
// CHECK:STDOUT:   %.e1e: ref %tuple.type.555 = temporary invalid, %tuple.479 [concrete]
// CHECK:STDOUT:   %tuple.type.f94: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %tuple.ad8: %tuple.type.f94 = tuple_value (%int_1.5b8, %int_2.ecc) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc17_32.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.e1e, %Destroy.Op.1a2547.3 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.777 = out_param_pattern [concrete = constants.%return.param_patt.789]
// CHECK:STDOUT:     %return.patt: %pattern_type.777 = return_slot_pattern %return.param_patt, %.loc5_30.2 [concrete = constants.%return.patt.e16]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc5_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %i32.loc5_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_30.1: %tuple.type.ff9 = tuple_literal (%i32.loc5_17, %i32.loc5_22, %i32.loc5_27) [concrete = constants.%tuple.04d]
// CHECK:STDOUT:     %.loc5_30.2: type = converted %.loc5_30.1, constants.%tuple.type.555 [concrete = constants.%tuple.type.555]
// CHECK:STDOUT:     %.loc5_30.3: Core.Form = init_form %.loc5_30.2 [concrete = constants.%.718]
// CHECK:STDOUT:     %return.param: ref %tuple.type.555 = out_param call_param0
// CHECK:STDOUT:     %return: ref %tuple.type.555 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %tuple.type.555 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc5_49.1: %tuple.type.37f = tuple_literal (%int_1, %int_2, %int_3) [concrete = constants.%tuple.2d5]
// CHECK:STDOUT:   %impl.elem0.loc5_49.1: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_49.1: <bound method> = bound_method %int_1, %impl.elem0.loc5_49.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.094]
// CHECK:STDOUT:   %specific_fn.loc5_49.1: <specific function> = specific_function %impl.elem0.loc5_49.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_49.2: <bound method> = bound_method %int_1, %specific_fn.loc5_49.1 [concrete = constants.%bound_method.953]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.1: init %i32 = call %bound_method.loc5_49.2(%int_1) [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %.loc5_49.2: init %i32 = converted %int_1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.1 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %tuple.elem0: ref %i32 = tuple_access %return.param, element0
// CHECK:STDOUT:   %.loc5_49.3: init %i32 to %tuple.elem0 = in_place_init %.loc5_49.2 [concrete = constants.%int_1.0c6]
// CHECK:STDOUT:   %impl.elem0.loc5_49.2: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_49.3: <bound method> = bound_method %int_2, %impl.elem0.loc5_49.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc5_49.2: <specific function> = specific_function %impl.elem0.loc5_49.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_49.4: <bound method> = bound_method %int_2, %specific_fn.loc5_49.2 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.2: init %i32 = call %bound_method.loc5_49.4(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc5_49.4: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.2 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %tuple.elem1: ref %i32 = tuple_access %return.param, element1
// CHECK:STDOUT:   %.loc5_49.5: init %i32 to %tuple.elem1 = in_place_init %.loc5_49.4 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %impl.elem0.loc5_49.3: %.b7a = impl_witness_access constants.%ImplicitAs.impl_witness.a2a, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc5_49.5: <bound method> = bound_method %int_3, %impl.elem0.loc5_49.3 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc5_49.3: <specific function> = specific_function %impl.elem0.loc5_49.3, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc5_49.6: <bound method> = bound_method %int_3, %specific_fn.loc5_49.3 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.3: init %i32 = call %bound_method.loc5_49.6(%int_3) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc5_49.6: init %i32 = converted %int_3, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc5_49.3 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %tuple.elem2: ref %i32 = tuple_access %return.param, element2
// CHECK:STDOUT:   %.loc5_49.7: init %i32 to %tuple.elem2 = in_place_init %.loc5_49.6 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc5_49.8: init %tuple.type.555 to %return.param = tuple_init (%.loc5_49.3, %.loc5_49.5, %.loc5_49.7) [concrete = constants.%tuple.479]
// CHECK:STDOUT:   %.loc5_50: init %tuple.type.555 = converted %.loc5_49.1, %.loc5_49.8 [concrete = constants.%tuple.479]
// CHECK:STDOUT:   return %.loc5_50 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.var: ref <error> = var_storage %a.var_patt [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc17_43: %tuple.type.f94 = tuple_literal (%int_1, %int_2) [concrete = constants.%tuple.ad8]
// CHECK:STDOUT:   assign %a.var, <error>
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a: <error> = wrapper_binding a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: <error> = ref_binding_pattern a [concrete = <error>]
// CHECK:STDOUT:     %a.var_patt: <error> = var_pattern %a.patt [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.e1e)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_32.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_32.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc17_32.3(%self.param: ref %tuple.type.555) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- return_in_place_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.115: ref %C = temporary invalid, %C.val [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%C.val) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc13_7.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.115, %Destroy.Op.1a2547.2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc6_16: Core.Form = init_form %C.ref [concrete = constants.%.a44]
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param0
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc6_28.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc6_28.2: init %C to %return.param = class_init () [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc6_29: init %C = converted %.loc6_28.1, %.loc6_28.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   return %.loc6_29 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc13_7.1: ref %C = temporary_storage
// CHECK:STDOUT:   %F.call: init %C to %.loc13_7.1 = call %F.ref() [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc13_7.2: ref %C = temporary %.loc13_7.1, %F.call [concrete = constants.%.115]
// CHECK:STDOUT:   %.loc13_7.3: %C = acquire_value %.loc13_7.2 [concrete = constants.%C.val]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%C.val) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %G.call: init %empty_tuple.type = call %G.specific_fn()
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.115)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_7.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc13_7.2(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:

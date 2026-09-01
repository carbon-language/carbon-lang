// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/member_access.carbon

// --- simple_member_access.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[template T: type](x: T) -> i32 {
  let n: i32 = x.n;
  return n;
}
//@dump-sem-ir-end

class C {
  var n: i32;
}

fn Test1(c: C) {
  F(c);
}

fn Test2(x: {.m: i32, .n: i32}) {
  F(x);
}

// --- fail_no_such_member.carbon

library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> i32 {
  let n: i32 = x.n;
  return n;
}

class D {
  var m: i32;
}

fn Test(d: D) {
  // CHECK:STDERR: fail_no_such_member.carbon:[[@LINE+7]]:3: error: unable to monomorphize specific `F(D)` [ResolvingSpecificHere]
  // CHECK:STDERR:   F(d);
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_no_such_member.carbon:[[@LINE-12]]:16: note: member name `n` not found in `D` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   let n: i32 = x.n;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR:
  F(d);
}

// --- fail_member_wrong_type.carbon

library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> i32 {
  let n: i32 = x.n;
  return n;
}

class E {
  class F {}
  var n: F;
}

fn Test(e: E) {
  // CHECK:STDERR: fail_member_wrong_type.carbon:[[@LINE+10]]:3: error: unable to monomorphize specific `F(E)` [ResolvingSpecificHere]
  // CHECK:STDERR:   F(e);
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_member_wrong_type.carbon:[[@LINE-13]]:16: note: cannot implicitly convert expression of type `F` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let n: i32 = x.n;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR: fail_member_wrong_type.carbon:[[@LINE-16]]:16: note: type `F` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let n: i32 = x.n;
  // CHECK:STDERR:                ^~~
  // CHECK:STDERR:
  F(e);
}

// --- fail_todo_compound_member_access.carbon

library "[[@TEST_NAME]]";

base class C {
  var n: i32;
}

//@dump-sem-ir-begin
fn F[template T: type](x: T) -> i32 {
  return x.(C.n);
}
//@dump-sem-ir-end

class D {
  extend base: C;
}

fn G(d: D) -> i32 {
    // CHECK:STDERR: fail_todo_compound_member_access.carbon:[[@LINE+7]]:12: error: unable to monomorphize specific `F(D)` [ResolvingSpecificHere]
    // CHECK:STDERR:     return F(d);
    // CHECK:STDERR:            ^
    // CHECK:STDERR: fail_todo_compound_member_access.carbon:[[@LINE-12]]:3: note: cannot access member of interface `Core.ImplicitAs(i32)` in type `i32` that does not implement that interface [MissingImplInMemberAccess]
    // CHECK:STDERR:   return x.(C.n);
    // CHECK:STDERR:   ^~~~~~~~~~~~~~~
    // CHECK:STDERR:
    return F(d);
}

// CHECK:STDOUT: --- simple_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47011.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %n.patt.c48: %pattern_type.6b6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %.b3e: %T.67db0b.1 = splice_inst @F.%.loc6_17.4 [template]
// CHECK:STDOUT:   %.432: @F.%.loc6_17.7 (@F.%.loc6_17.7) = splice_inst @F.%.loc6_17.6 [template]
// CHECK:STDOUT:   %.43a: %i32 = splice_inst @F.%.loc6_17.9 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.b51: <witness> = impl_witness imports.%Copy.impl_witness_table.8d2, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.b51) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.381: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.737: type = fn_type_with_self_type %Copy.WithSelf.Op.type.381, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %struct_type.m.n: type = struct_type {.m: %i32, .n: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %struct_type.m.n [concrete]
// CHECK:STDOUT:   %x.param_patt.c8a: %pattern_type.860 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.3a7: %pattern_type.860 = wrapper_binding_pattern x, %x.param_patt.c8a [concrete]
// CHECK:STDOUT:   %inst.as_compatible.d98: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f8c: %C = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.1c3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ba1: %i32 = splice_block %.1e9 {
// CHECK:STDOUT:       %n.ref: %C.elem = name_ref n, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:       %.a48: ref %i32 = class_element_access %.f8c, element0
// CHECK:STDOUT:       %.1e9: %i32 = acquire_value %.a48
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.4ec: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.0b7: %i32 = splice_block %.ba1 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.08f: <witness> = complete_type_witness %struct_type.m.n [concrete]
// CHECK:STDOUT:   %inst.as_compatible.bba: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a89: %struct_type.m.n = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.struct_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ebd: %i32 = struct_access %.a89, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.1b4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.214: %i32 = splice_block %.ebd {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc5_25.1 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_33: Core.Form = init_form %i32.loc5 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.%T.loc5_16.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @F.%T.loc5_16.1 (%T.67db0b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_16.1 [template = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc5_25.2 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc6_17.4: <instruction> = refine_type_action %x.ref, %T.loc5_16.1 [template]
// CHECK:STDOUT:   %.loc6_17.5: @F.%T.loc5_16.1 (%T.67db0b.1) = splice_inst %.loc6_17.4 [template = %.loc6_17.5 (constants.%.b3e)]
// CHECK:STDOUT:   %.loc6_17.6: <instruction> = access_member_action %.loc6_17.1, n [template]
// CHECK:STDOUT:   %.loc6_17.7: type = type_of_inst %.loc6_17.6 [template]
// CHECK:STDOUT:   %.loc6_17.8: @F.%.loc6_17.7 (@F.%.loc6_17.7) = splice_inst %.loc6_17.6 [template = %.loc6_17.8 (constants.%.432)]
// CHECK:STDOUT:   %.loc6_17.9: <instruction> = convert_to_value_action %.loc6_17.2, constants.%i32 [template]
// CHECK:STDOUT:   %.loc6_17.10: %i32 = splice_inst %.loc6_17.9 [template = %.loc6_17.10 (constants.%.43a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_17.1: @F.%T.loc5_16.1 (%T.67db0b.1) = splice_inst %.loc6_17.4 [template = %.loc6_17.5 (constants.%.b3e)]
// CHECK:STDOUT:     %.loc6_17.2: @F.%.loc6_17.7 (@F.%.loc6_17.7) = splice_inst %.loc6_17.6 [template = %.loc6_17.8 (constants.%.432)]
// CHECK:STDOUT:     %.loc6_17.3: %i32 = splice_inst %.loc6_17.9 [template = %.loc6_17.10 (constants.%.43a)]
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %.loc6_17.3
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: %pattern_type.6b6 = value_binding_pattern n [concrete = constants.%n.patt.c48]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0: %.737 = impl_witness_access constants.%Copy.impl_witness.b51, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:     %bound_method.loc7_10.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_10.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_10.2(%n.ref)
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.953
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cdf
// CHECK:STDOUT:   %.loc6_17.4 => constants.%inst.as_compatible.d98
// CHECK:STDOUT:   %.loc6_17.5 => invalid
// CHECK:STDOUT:   %.loc6_17.6 => constants.%inst.splice_block.1c3
// CHECK:STDOUT:   %.loc6_17.7 => constants.%i32
// CHECK:STDOUT:   %.loc6_17.8 => invalid
// CHECK:STDOUT:   %.loc6_17.9 => constants.%inst.splice_block.4ec
// CHECK:STDOUT:   %.loc6_17.10 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct_type.m.n) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%struct_type.m.n
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.860
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.c8a
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.3a7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.08f
// CHECK:STDOUT:   %.loc6_17.4 => constants.%inst.as_compatible.bba
// CHECK:STDOUT:   %.loc6_17.5 => invalid
// CHECK:STDOUT:   %.loc6_17.6 => constants.%inst.struct_access
// CHECK:STDOUT:   %.loc6_17.7 => constants.%i32
// CHECK:STDOUT:   %.loc6_17.8 => invalid
// CHECK:STDOUT:   %.loc6_17.9 => constants.%inst.splice_block.1b4
// CHECK:STDOUT:   %.loc6_17.10 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %.b3e: %T = splice_inst @F.%.loc10_11.3 [template]
// CHECK:STDOUT:   %.244: @F.%.loc10_11.6 (@F.%.loc10_11.6) = splice_inst @F.%.loc10_11.5 [template]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.294: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.945: %ImplicitAs.WithSelf.Convert.type.97e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.368: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %.841: @F.%.loc10_17.6 (@F.%.loc10_17.6) = splice_inst @F.%.loc10_17.5 [template]
// CHECK:STDOUT:   %.cbd: @F.%.loc10_17.9 (@F.%.loc10_17.9) = splice_inst @F.%.loc10_17.8 [template]
// CHECK:STDOUT:   %.c63: %i32 = splice_inst @F.%.loc10_17.11 [template]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.base.085: type = struct_type {.base: %C} [concrete]
// CHECK:STDOUT:   %complete_type.756: <witness> = complete_type_witness %struct_type.base.085 [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %x.param_patt.23b: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c51: %pattern_type.d8d = wrapper_binding_pattern x, %x.param_patt.23b [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.602: %D = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.08b: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.1843e: ref %i32 = splice_block %.a6d {
// CHECK:STDOUT:       %.68a: %C = as_compatible %.602
// CHECK:STDOUT:       %.c3b: ref %C = class_element_access %.68a, element0
// CHECK:STDOUT:       %.59a: ref %C = converted %.602, %.c3b
// CHECK:STDOUT:       %.a6d: ref %i32 = class_element_access %.59a, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.edd: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.432: <error> = splice_block <error> [concrete = <error>] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97e) = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.945)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_16.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc9_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc9_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc9_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc9_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc9_25.1 [template = %x.patt.loc9_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_33: Core.Form = init_form %i32 [concrete = constants.%.795f]
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_16.2: type = symbolic_binding T, 0, template [template = %T.loc9_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc9_16.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_16.2 [template = %T.loc9_16.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc9_16.1 (%T) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_16.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc9_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc9_16.1: type = symbolic_binding T, 0, template [template = %T.loc9_16.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc9_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc9_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc9_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc9_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc9_25.2 [template = %x.patt.loc9_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc9_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc10_11.3: <instruction> = refine_type_action %x.ref, %T.loc9_16.1 [template]
// CHECK:STDOUT:   %.loc10_11.4: @F.%T.loc9_16.1 (%T) = splice_inst %.loc10_11.3 [template = %.loc10_11.4 (constants.%.b3e)]
// CHECK:STDOUT:   %.loc10_11.5: <instruction> = compound_member_access_action %.loc10_11.1, %n.ref [template]
// CHECK:STDOUT:   %.loc10_11.6: type = type_of_inst %.loc10_11.5 [template]
// CHECK:STDOUT:   %.loc10_11.7: @F.%.loc10_11.6 (@F.%.loc10_11.6) = splice_inst %.loc10_11.5 [template = %.loc10_11.7 (constants.%.244)]
// CHECK:STDOUT:   %.loc10_17.5: <instruction> = compound_member_access_action %.loc10_11.2, constants.%assoc0.368 [template]
// CHECK:STDOUT:   %.loc10_17.6: type = type_of_inst %.loc10_17.5 [template]
// CHECK:STDOUT:   %.loc10_17.7: @F.%.loc10_17.6 (@F.%.loc10_17.6) = splice_inst %.loc10_17.5 [template = %.loc10_17.7 (constants.%.841)]
// CHECK:STDOUT:   %.loc10_17.8: <instruction> = call_action (%.loc10_17.1), true [template]
// CHECK:STDOUT:   %.loc10_17.9: type = type_of_inst %.loc10_17.8 [template]
// CHECK:STDOUT:   %.loc10_17.10: @F.%.loc10_17.9 (@F.%.loc10_17.9) = splice_inst %.loc10_17.8 [template = %.loc10_17.10 (constants.%.cbd)]
// CHECK:STDOUT:   %.loc10_17.11: <instruction> = convert_to_category_action %.loc10_17.3, element10 [template]
// CHECK:STDOUT:   %.loc10_17.12: %i32 = splice_inst %.loc10_17.11 [template = %.loc10_17.12 (constants.%.c63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc9_16.1 (%T)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc9_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %n.ref: %C.elem = name_ref n, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:     %.loc10_11.1: @F.%T.loc9_16.1 (%T) = splice_inst %.loc10_11.3 [template = %.loc10_11.4 (constants.%.b3e)]
// CHECK:STDOUT:     %.loc10_11.2: @F.%.loc10_11.6 (@F.%.loc10_11.6) = splice_inst %.loc10_11.5 [template = %.loc10_11.7 (constants.%.244)]
// CHECK:STDOUT:     %.loc10_17.1: @F.%.loc10_17.6 (@F.%.loc10_17.6) = splice_inst %.loc10_17.5 [template = %.loc10_17.7 (constants.%.841)]
// CHECK:STDOUT:     %.loc10_17.2: @F.%.loc10_17.9 (@F.%.loc10_17.9) = splice_inst %.loc10_17.8 [template = %.loc10_17.10 (constants.%.cbd)]
// CHECK:STDOUT:     %.loc10_17.3: %i32 = converted %.loc10_11.2, %.loc10_17.2 [template = %.loc10_17.10 (constants.%.cbd)]
// CHECK:STDOUT:     %.loc10_17.4: %i32 = splice_inst %.loc10_17.11 [template = %.loc10_17.12 (constants.%.c63)]
// CHECK:STDOUT:     return %.loc10_17.4
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc9_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_16.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc9_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc9_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.patt.loc9_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc9_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d8d
// CHECK:STDOUT:   %x.param_patt.loc9_25.2 => constants.%x.param_patt.23b
// CHECK:STDOUT:   %x.patt.loc9_25.2 => constants.%x.patt.c51
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.756
// CHECK:STDOUT:   %.loc10_11.3 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.loc10_11.4 => invalid
// CHECK:STDOUT:   %.loc10_11.5 => constants.%inst.splice_block.08b
// CHECK:STDOUT:   %.loc10_11.6 => constants.%i32
// CHECK:STDOUT:   %.loc10_11.7 => invalid
// CHECK:STDOUT:   %.loc10_17.5 => constants.%inst.splice_block.edd
// CHECK:STDOUT:   %.loc10_17.6 => <error>
// CHECK:STDOUT:   %.loc10_17.7 => <error>
// CHECK:STDOUT:   %.loc10_17.8 => <error>
// CHECK:STDOUT:   %.loc10_17.9 => <error>
// CHECK:STDOUT:   %.loc10_17.10 => <error>
// CHECK:STDOUT:   %.loc10_17.11 => <error>
// CHECK:STDOUT:   %.loc10_17.12 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

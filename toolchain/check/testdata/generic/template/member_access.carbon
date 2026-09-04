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
// CHECK:STDOUT:   %n.patt: %pattern_type.6b6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %.098: %T.67db0b.1 = splice_inst @F.%.loc6_17.5 [template]
// CHECK:STDOUT:   %.250: type = type_of_inst @F.%.loc6_17.7 [template]
// CHECK:STDOUT:   %.7a1: %.250 = splice_inst @F.%.loc6_17.7 [template]
// CHECK:STDOUT:   %.ae0: %i32 = splice_inst @F.%.loc6_17.10 [template]
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
// CHECK:STDOUT:   %inst.specific_inst.669: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.bfd: %C = specific_inst @F.%x.ref, @F(%C)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.62c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b57: %i32 = splice_block %.232 {
// CHECK:STDOUT:       %n.ref: %C.elem = name_ref n, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:       %.d33: ref %i32 = class_element_access %.bfd, element0
// CHECK:STDOUT:       %.232: %i32 = acquire_value %.d33
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.575: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.51a: %i32 = splice_block %.b57 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.08f: <witness> = complete_type_witness %struct_type.m.n [concrete]
// CHECK:STDOUT:   %inst.specific_inst.7c9: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d1d: %struct_type.m.n = specific_inst @F.%x.ref, @F(%struct_type.m.n)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.struct_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.465: %i32 = struct_access %.d1d, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.177: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.605: %i32 = splice_block %.465 {}
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
// CHECK:STDOUT:   %.loc6_17.5: <instruction> = refine_inst_action %x.ref [template]
// CHECK:STDOUT:   %.loc6_17.6: @F.%T.loc5_16.1 (%T.67db0b.1) = splice_inst %.loc6_17.5 [template = %.loc6_17.6 (constants.%.098)]
// CHECK:STDOUT:   %.loc6_17.7: <instruction> = access_member_action %.loc6_17.1, n [template]
// CHECK:STDOUT:   %.loc6_17.8: type = type_of_inst %.loc6_17.7 [template = %.loc6_17.8 (constants.%.250)]
// CHECK:STDOUT:   %.loc6_17.9: @F.%.loc6_17.8 (%.250) = splice_inst %.loc6_17.7 [template = %.loc6_17.9 (constants.%.7a1)]
// CHECK:STDOUT:   %.loc6_17.10: <instruction> = convert_to_value_action %.loc6_17.3, constants.%i32 [template]
// CHECK:STDOUT:   %.loc6_17.11: %i32 = splice_inst %.loc6_17.10 [template = %.loc6_17.11 (constants.%.ae0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_17.1: @F.%T.loc5_16.1 (%T.67db0b.1) = splice_inst %.loc6_17.5 [template = %.loc6_17.6 (constants.%.098)]
// CHECK:STDOUT:     %.loc6_17.2: type = type_of_inst %.loc6_17.7 [template = %.loc6_17.8 (constants.%.250)]
// CHECK:STDOUT:     %.loc6_17.3: @F.%.loc6_17.8 (%.250) = splice_inst %.loc6_17.7 [template = %.loc6_17.9 (constants.%.7a1)]
// CHECK:STDOUT:     %.loc6_17.4: %i32 = splice_inst %.loc6_17.10 [template = %.loc6_17.11 (constants.%.ae0)]
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %.loc6_17.4
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: %pattern_type.6b6 = value_binding_pattern n [concrete = constants.%n.patt]
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
// CHECK:STDOUT:   %.loc6_17.5 => constants.%inst.specific_inst.669
// CHECK:STDOUT:   %.loc6_17.6 => invalid
// CHECK:STDOUT:   %.loc6_17.7 => constants.%inst.splice_block.62c
// CHECK:STDOUT:   %.loc6_17.8 => constants.%i32
// CHECK:STDOUT:   %.loc6_17.9 => invalid
// CHECK:STDOUT:   %.loc6_17.10 => constants.%inst.splice_block.575
// CHECK:STDOUT:   %.loc6_17.11 => invalid
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
// CHECK:STDOUT:   %.loc6_17.5 => constants.%inst.specific_inst.7c9
// CHECK:STDOUT:   %.loc6_17.6 => invalid
// CHECK:STDOUT:   %.loc6_17.7 => constants.%inst.struct_access
// CHECK:STDOUT:   %.loc6_17.8 => constants.%i32
// CHECK:STDOUT:   %.loc6_17.9 => invalid
// CHECK:STDOUT:   %.loc6_17.10 => constants.%inst.splice_block.177
// CHECK:STDOUT:   %.loc6_17.11 => invalid
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
// CHECK:STDOUT:   %.098: %T = splice_inst @F.%.loc10_11.4 [template]
// CHECK:STDOUT:   %.862: type = type_of_inst @F.%.loc10_11.6 [template]
// CHECK:STDOUT:   %.942: %.862 = splice_inst @F.%.loc10_11.6 [template]
// CHECK:STDOUT:   %Dest: type = symbolic_binding Dest, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%Dest)> [symbolic]
// CHECK:STDOUT:   %Self.294: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Dest, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.945: %ImplicitAs.WithSelf.Convert.type.97e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e23: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %assoc0.368: %ImplicitAs.assoc_type.e23 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %.da7: type = type_of_inst @F.%.loc10_17.8 [template]
// CHECK:STDOUT:   %.823: %.da7 = splice_inst @F.%.loc10_17.8 [template]
// CHECK:STDOUT:   %.45b: type = type_of_inst @F.%.loc10_17.11 [template]
// CHECK:STDOUT:   %.497: %.45b = splice_inst @F.%.loc10_17.11 [template]
// CHECK:STDOUT:   %.572: %i32 = splice_inst @F.%.loc10_17.14 [template]
// CHECK:STDOUT:   %.829: %i32 = splice_inst @F.%.loc10_17.16 [template]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.base.085: type = struct_type {.base: %C} [concrete]
// CHECK:STDOUT:   %complete_type.756: <witness> = complete_type_witness %struct_type.base.085 [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %x.param_patt.23b: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c51: %pattern_type.d8d = wrapper_binding_pattern x, %x.param_patt.23b [concrete]
// CHECK:STDOUT:   %inst.specific_inst: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.24c: %D = specific_inst @F.%x.ref, @F(%D)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.edc: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.859: ref %i32 = splice_block %.b2f {
// CHECK:STDOUT:       %.179: %C = as_compatible %.24c
// CHECK:STDOUT:       %.8a5: ref %C = class_element_access %.179, element0
// CHECK:STDOUT:       %.313: ref %C = converted %.24c, %.8a5
// CHECK:STDOUT:       %.b2f: ref %i32 = class_element_access %.313, element0
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
// CHECK:STDOUT:   %.loc10_11.4: <instruction> = refine_inst_action %x.ref [template]
// CHECK:STDOUT:   %.loc10_11.5: @F.%T.loc9_16.1 (%T) = splice_inst %.loc10_11.4 [template = %.loc10_11.5 (constants.%.098)]
// CHECK:STDOUT:   %.loc10_11.6: <instruction> = compound_member_access_action %.loc10_11.1, %n.ref [template]
// CHECK:STDOUT:   %.loc10_11.7: type = type_of_inst %.loc10_11.6 [template = %.loc10_11.7 (constants.%.862)]
// CHECK:STDOUT:   %.loc10_11.8: @F.%.loc10_11.7 (%.862) = splice_inst %.loc10_11.6 [template = %.loc10_11.8 (constants.%.942)]
// CHECK:STDOUT:   %.loc10_17.8: <instruction> = compound_member_access_action %.loc10_11.3, constants.%assoc0.368 [template]
// CHECK:STDOUT:   %.loc10_17.9: type = type_of_inst %.loc10_17.8 [template = %.loc10_17.9 (constants.%.da7)]
// CHECK:STDOUT:   %.loc10_17.10: @F.%.loc10_17.9 (%.da7) = splice_inst %.loc10_17.8 [template = %.loc10_17.10 (constants.%.823)]
// CHECK:STDOUT:   %.loc10_17.11: <instruction> = call_action (%.loc10_17.2), true [template]
// CHECK:STDOUT:   %.loc10_17.12: type = type_of_inst %.loc10_17.11 [template = %.loc10_17.12 (constants.%.45b)]
// CHECK:STDOUT:   %.loc10_17.13: @F.%.loc10_17.12 (%.45b) = splice_inst %.loc10_17.11 [template = %.loc10_17.13 (constants.%.497)]
// CHECK:STDOUT:   %.loc10_17.14: <instruction> = refine_inst_action %.loc10_17.5 [template]
// CHECK:STDOUT:   %.loc10_17.15: %i32 = splice_inst %.loc10_17.14 [template = %.loc10_17.15 (constants.%.572)]
// CHECK:STDOUT:   %.loc10_17.16: <instruction> = convert_to_category_action %.loc10_17.6, element10 [template]
// CHECK:STDOUT:   %.loc10_17.17: %i32 = splice_inst %.loc10_17.16 [template = %.loc10_17.17 (constants.%.829)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc9_16.1 (%T)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc9_16.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %n.ref: %C.elem = name_ref n, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:     %.loc10_11.1: @F.%T.loc9_16.1 (%T) = splice_inst %.loc10_11.4 [template = %.loc10_11.5 (constants.%.098)]
// CHECK:STDOUT:     %.loc10_11.2: type = type_of_inst %.loc10_11.6 [template = %.loc10_11.7 (constants.%.862)]
// CHECK:STDOUT:     %.loc10_11.3: @F.%.loc10_11.7 (%.862) = splice_inst %.loc10_11.6 [template = %.loc10_11.8 (constants.%.942)]
// CHECK:STDOUT:     %.loc10_17.1: type = type_of_inst %.loc10_17.8 [template = %.loc10_17.9 (constants.%.da7)]
// CHECK:STDOUT:     %.loc10_17.2: @F.%.loc10_17.9 (%.da7) = splice_inst %.loc10_17.8 [template = %.loc10_17.10 (constants.%.823)]
// CHECK:STDOUT:     %.loc10_17.3: type = type_of_inst %.loc10_17.11 [template = %.loc10_17.12 (constants.%.45b)]
// CHECK:STDOUT:     %.loc10_17.4: @F.%.loc10_17.12 (%.45b) = splice_inst %.loc10_17.11 [template = %.loc10_17.13 (constants.%.497)]
// CHECK:STDOUT:     %.loc10_17.5: %i32 = converted %.loc10_11.3, %.loc10_17.4 [template = %.loc10_17.13 (constants.%.497)]
// CHECK:STDOUT:     %.loc10_17.6: %i32 = splice_inst %.loc10_17.14 [template = %.loc10_17.15 (constants.%.572)]
// CHECK:STDOUT:     %.loc10_17.7: %i32 = splice_inst %.loc10_17.16 [template = %.loc10_17.17 (constants.%.829)]
// CHECK:STDOUT:     return %.loc10_17.7
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
// CHECK:STDOUT:   %.loc10_11.4 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc10_11.5 => invalid
// CHECK:STDOUT:   %.loc10_11.6 => constants.%inst.splice_block.edc
// CHECK:STDOUT:   %.loc10_11.7 => constants.%i32
// CHECK:STDOUT:   %.loc10_11.8 => invalid
// CHECK:STDOUT:   %.loc10_17.8 => constants.%inst.splice_block.edd
// CHECK:STDOUT:   %.loc10_17.9 => <error>
// CHECK:STDOUT:   %.loc10_17.10 => <error>
// CHECK:STDOUT:   %.loc10_17.11 => <error>
// CHECK:STDOUT:   %.loc10_17.12 => <error>
// CHECK:STDOUT:   %.loc10_17.13 => <error>
// CHECK:STDOUT:   %.loc10_17.14 => <error>
// CHECK:STDOUT:   %.loc10_17.15 => <error>
// CHECK:STDOUT:   %.loc10_17.16 => <error>
// CHECK:STDOUT:   %.loc10_17.17 => <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:

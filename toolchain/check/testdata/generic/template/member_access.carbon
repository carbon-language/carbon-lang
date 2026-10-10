// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
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

// --- compound_member_access.carbon

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
    return F(d);
}

// --- protected_member_access.carbon

library "[[@TEST_NAME]]";

base class A {
  protected var n: i32 = 1;
}

class B(template T: type) {
  extend base: A;

  fn F(self) -> i32 {
    return self.n;
  }
}

fn G() {
  let b: B(i32) = {.base = {}};
  b.F();
}

// CHECK:STDOUT: --- simple_member_access.carbon
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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.973: %pattern_type.df9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a21: %pattern_type.df9 = return_slot_pattern %return.param_patt.973, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %n.patt: %pattern_type.df9 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %.237: type = type_of_inst @F.%.loc6_17.4 [template]
// CHECK:STDOUT:   %.f96: %.237 = splice_inst @F.%.loc6_17.4 [template]
// CHECK:STDOUT:   %.73f: %i32 = splice_inst @F.%.loc6_17.7 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.3ab: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %struct_type.m.n: type = struct_type {.m: %i32, .n: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.a2e: type = pattern_type %struct_type.m.n [concrete]
// CHECK:STDOUT:   %x.param_patt.98b: %pattern_type.a2e = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.bcd: %pattern_type.a2e = wrapper_binding_pattern x, %x.param_patt.98b [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c46: %i32 = splice_block %.dd6 {
// CHECK:STDOUT:       %.bfd: %C = specific_inst @F.%x.ref, @F(%C)
// CHECK:STDOUT:       %n.ref: %C.elem = name_ref n, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:       %.b6f: ref %i32 = class_element_access %.bfd, element0
// CHECK:STDOUT:       %.dd6: %i32 = acquire_value %.b6f
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a24: %i32 = splice_block %.c46 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.69e: <witness> = complete_type_witness %struct_type.m.n [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a01: %i32 = splice_block %.b50 {
// CHECK:STDOUT:       %.dfd: %struct_type.m.n = specific_inst @F.%x.ref, @F(%struct_type.m.n)
// CHECK:STDOUT:       %.b50: %i32 = struct_access %.dfd, element1
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a8d: %i32 = splice_block %.a01 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc5_25.1 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.df9 = out_param_pattern [concrete = constants.%return.param_patt.973]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.a21]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_33: Core.Form = init_form %i32.loc5 [concrete = constants.%.4e1]
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
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
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_16.1 [template = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc5_25.2 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc6_17.4: <instruction> = access_member_action %x.ref, n [template]
// CHECK:STDOUT:   %.loc6_17.5: type = type_of_inst %.loc6_17.4 [template = %.loc6_17.5 (constants.%.237)]
// CHECK:STDOUT:   %.loc6_17.6: @F.%.loc6_17.5 (%.237) = splice_inst %.loc6_17.4 [template = %.loc6_17.6 (constants.%.f96)]
// CHECK:STDOUT:   %.loc6_17.7: <instruction> = convert_to_value_action %.loc6_17.2, constants.%i32 [template]
// CHECK:STDOUT:   %.loc6_17.8: %i32 = splice_inst %.loc6_17.7 [template = %.loc6_17.8 (constants.%.73f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_17.1: type = type_of_inst %.loc6_17.4 [template = %.loc6_17.5 (constants.%.237)]
// CHECK:STDOUT:     %.loc6_17.2: @F.%.loc6_17.5 (%.237) = splice_inst %.loc6_17.4 [template = %.loc6_17.6 (constants.%.f96)]
// CHECK:STDOUT:     %.loc6_17.3: %i32 = splice_inst %.loc6_17.7 [template = %.loc6_17.8 (constants.%.73f)]
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %.loc6_17.3
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: %pattern_type.df9 = value_binding_pattern n [concrete = constants.%n.patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:     %bound_method.loc7_10.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc7_10.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_10.2(%n.ref)
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.953
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3ab
// CHECK:STDOUT:   %.loc6_17.4 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_17.5 => constants.%i32
// CHECK:STDOUT:   %.loc6_17.6 => <not constant>
// CHECK:STDOUT:   %.loc6_17.7 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_17.8 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct_type.m.n) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%struct_type.m.n
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.a2e
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.98b
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.bcd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.69e
// CHECK:STDOUT:   %.loc6_17.4 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_17.5 => constants.%i32
// CHECK:STDOUT:   %.loc6_17.6 => <not constant>
// CHECK:STDOUT:   %.loc6_17.7 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc6_17.8 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- compound_member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.973: %pattern_type.df9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a21: %pattern_type.df9 = return_slot_pattern %return.param_patt.973, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc10_11.3 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc10_11.3 [template]
// CHECK:STDOUT:   %.cac: %i32 = splice_inst @F.%.loc10_17.2 [template]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %struct_type.base.085: type = struct_type {.base: %C} [concrete]
// CHECK:STDOUT:   %complete_type.756: <witness> = complete_type_witness %struct_type.base.085 [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %x.param_patt.23b: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c51: %pattern_type.d8d = wrapper_binding_pattern x, %x.param_patt.23b [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.221: ref %i32 = splice_block %.9d9 {
// CHECK:STDOUT:       %.24c: %D = specific_inst @F.%x.ref, @F(%D)
// CHECK:STDOUT:       %.ff4: ref %C = class_element_access %.24c, element0
// CHECK:STDOUT:       %.07e: ref %C = converted %.24c, %.ff4
// CHECK:STDOUT:       %.9d9: ref %i32 = class_element_access %.07e, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d69: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:       %.eb2: %i32 = acquire_value %.221
// CHECK:STDOUT:       %impl.elem0.0a9: %.187 = impl_witness_access %Copy.impl_witness.4c3, element0 [concrete = %Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:       %bound_method.990: <bound method> = bound_method %.eb2, %impl.elem0.0a9
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.0a9, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.6df: <bound method> = bound_method %.eb2, %specific_fn
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.6df(%.eb2)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/types/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc9_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc9_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc9_25.1: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc9_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc9_25.1: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc9_25.1 [template = %x.patt.loc9_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.df9 = out_param_pattern [concrete = constants.%return.param_patt.973]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a21]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc9_33: Core.Form = init_form %i32 [concrete = constants.%.4e1]
// CHECK:STDOUT:     %.loc9_18.1: type = splice_block %.loc9_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_16.2: type = symbolic_binding T, 0, template [template = %T.loc9_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.%T.loc9_16.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_16.2 [template = %T.loc9_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @F.%T.loc9_16.1 (%T.67db0b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc9_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc9_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc9_16.1: type = symbolic_binding T, 0, template [template = %T.loc9_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc9_16.1 [template = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc9_25.2: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [template = %x.param_patt.loc9_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc9_25.2: @F.%pattern_type (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc9_25.2 [template = %x.patt.loc9_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc9_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc10_11.3: <instruction> = compound_member_access_action %x.ref, %n.ref [template]
// CHECK:STDOUT:   %.loc10_11.4: type = type_of_inst %.loc10_11.3 [template = %.loc10_11.4 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc10_11.5: @F.%.loc10_11.4 (%.8cb) = splice_inst %.loc10_11.3 [template = %.loc10_11.5 (constants.%.c23)]
// CHECK:STDOUT:   %.loc10_17.2: <instruction> = convert_action %.loc10_11.2, constants.%i32, element10 [template]
// CHECK:STDOUT:   %.loc10_17.3: %i32 = splice_inst %.loc10_17.2 [template = %.loc10_17.3 (constants.%.cac)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc9_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc9_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %n.ref: %C.elem = name_ref n, @C.%field_decl [concrete = @C.%field_decl]
// CHECK:STDOUT:     %.loc10_11.1: type = type_of_inst %.loc10_11.3 [template = %.loc10_11.4 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc10_11.2: @F.%.loc10_11.4 (%.8cb) = splice_inst %.loc10_11.3 [template = %.loc10_11.5 (constants.%.c23)]
// CHECK:STDOUT:     %.loc10_17.1: %i32 = splice_inst %.loc10_17.2 [template = %.loc10_17.3 (constants.%.cac)]
// CHECK:STDOUT:     return %.loc10_17.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc9_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc9_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:   %x.param_patt.loc9_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc9_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.patt.loc9_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc9_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d8d
// CHECK:STDOUT:   %x.param_patt.loc9_25.2 => constants.%x.param_patt.23b
// CHECK:STDOUT:   %x.patt.loc9_25.2 => constants.%x.patt.c51
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.756
// CHECK:STDOUT:   %.loc10_11.3 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc10_11.4 => constants.%i32
// CHECK:STDOUT:   %.loc10_11.5 => <not constant>
// CHECK:STDOUT:   %.loc10_17.2 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc10_17.3 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:

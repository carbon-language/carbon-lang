// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/member_access.carbon

// --- member_access.carbon

library "[[@TEST_NAME]]";

fn F[template T:! type](x: T) -> i32 {
  let n: i32 = x.n;
  return n;
}

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

fn F[template T:! type](x: T) -> i32 {
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

fn F[template T:! type](x: T) -> i32 {
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

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %Test1.type: type = fn_type @Test1 [concrete]
// CHECK:STDOUT:   %Test1: %Test1.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.e0d: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %struct_type.m.n: type = struct_type {.m: %i32, .n: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %struct_type.m.n [concrete]
// CHECK:STDOUT:   %Test2.type: type = fn_type @Test2 [concrete]
// CHECK:STDOUT:   %Test2: %Test2.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.4da: <specific function> = specific_function %F, @F(%struct_type.m.n) [concrete]
// CHECK:STDOUT:   %inst.as_compatible.ea0: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.bcf: %C = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.738: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.556: %i32 = splice_block %.33b {
// CHECK:STDOUT:       %n.ref: %C.elem = name_ref n, @C.%.loc10 [concrete = @C.%.loc10]
// CHECK:STDOUT:       %.cb0: ref %i32 = class_element_access %.bcf, element0
// CHECK:STDOUT:       %.33b: %i32 = acquire_value %.cb0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.3cc: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.98d: %i32 = splice_block %.556 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.08f: <witness> = complete_type_witness %struct_type.m.n [concrete]
// CHECK:STDOUT:   %inst.as_compatible.8b7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.63c: %struct_type.m.n = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.struct_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.29c: %i32 = struct_access %.63c, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.a6c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.110: %i32 = splice_block %.29c {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Test1 = %Test1.decl
// CHECK:STDOUT:     .Test2 = %Test2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.51d) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc4 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc4: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_34: Core.Form = init_form %i32.loc4 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_16.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @F.%T.loc4_16.1 (%T.67db0b.1) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Test1.decl: %Test1.type = fn_decl @Test1 [concrete = constants.%Test1] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.98b = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Test2.decl: %Test2.type = fn_decl @Test2 [concrete = constants.%Test2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.860 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.860 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %struct_type.m.n = value_param call_param0
// CHECK:STDOUT:     %.loc17: type = splice_block %struct_type.m.n [concrete = constants.%struct_type.m.n] {
// CHECK:STDOUT:       %i32.loc17_18: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %i32.loc17_27: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:       %struct_type.m.n: type = struct_type {.m: %i32, .n: %i32} [concrete = constants.%struct_type.m.n]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %struct_type.m.n = value_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc10: %C.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n [concrete = constants.%complete_type.cdf]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .n = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_16.1 [template = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc5_17.4: <instruction> = refine_type_action %x.ref, %T.loc4_16.1 [template]
// CHECK:STDOUT:   %.loc5_17.5: <instruction> = access_member_action %.loc5_17.1, n [template]
// CHECK:STDOUT:   %.loc5_17.6: type = type_of_inst %.loc5_17.5 [template]
// CHECK:STDOUT:   %.loc5_17.7: <instruction> = convert_to_value_action %.loc5_17.2, constants.%i32 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc4_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: %pattern_type.6b6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc5_17.1: @F.%T.loc4_16.1 (%T.67db0b.1) = splice_inst %.loc5_17.4
// CHECK:STDOUT:     %.loc5_17.2: @F.%.loc5_17.6 (@F.%.loc5_17.6) = splice_inst %.loc5_17.5
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_17.3: %i32 = splice_inst %.loc5_17.7
// CHECK:STDOUT:     %n: %i32 = value_binding n, %.loc5_17.3
// CHECK:STDOUT:     %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_10.2(%n.ref)
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test1(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.e0d]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%c.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test2(%x.param: %struct_type.m.n) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %x.ref: %struct_type.m.n = name_ref x, %x
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%struct_type.m.n) [concrete = constants.%F.specific_fn.4da]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%x.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cdf
// CHECK:STDOUT:   %.loc5_17.4 => constants.%inst.as_compatible.ea0
// CHECK:STDOUT:   %.loc5_17.5 => constants.%inst.splice_block.738
// CHECK:STDOUT:   %.loc5_17.6 => constants.%i32
// CHECK:STDOUT:   %.loc5_17.7 => constants.%inst.splice_block.3cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%struct_type.m.n) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%struct_type.m.n
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.860
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.08f
// CHECK:STDOUT:   %.loc5_17.4 => constants.%inst.as_compatible.8b7
// CHECK:STDOUT:   %.loc5_17.5 => constants.%inst.struct_access
// CHECK:STDOUT:   %.loc5_17.6 => constants.%i32
// CHECK:STDOUT:   %.loc5_17.7 => constants.%inst.splice_block.a6c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_no_such_member.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %D.elem: type = unbound_element_type %D, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.m: type = struct_type {.m: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.37d: <witness> = complete_type_witness %struct_type.m [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%D) [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.af0: %D = as_compatible @F.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %n.ref: <error> = name_ref n, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.432: <error> = splice_block <error> [concrete = <error>] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.51d) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc4 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc4: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_34: Core.Form = init_form %i32.loc4 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.%T.loc4_16.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @F.%T.loc4_16.1 (%T.67db0b.1) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type.d8d = value_param_pattern [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.d8d = at_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %d.param: %D = value_param call_param0
// CHECK:STDOUT:     %D.ref: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %d: %D = value_binding d, %d.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc10: %D.elem = field_decl m, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.m [concrete = constants.%complete_type.37d]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT:   .m = %.loc10
// CHECK:STDOUT:   .n = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_16.1 [template = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc5_17.4: <instruction> = refine_type_action %x.ref, %T.loc4_16.1 [template]
// CHECK:STDOUT:   %.loc5_17.5: <instruction> = access_member_action %.loc5_17.1, n [template]
// CHECK:STDOUT:   %.loc5_17.6: type = type_of_inst %.loc5_17.5 [template]
// CHECK:STDOUT:   %.loc5_17.7: <instruction> = convert_to_value_action %.loc5_17.2, constants.%i32 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc4_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: %pattern_type.6b6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.ref: @F.%T.loc4_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc5_17.1: @F.%T.loc4_16.1 (%T.67db0b.1) = splice_inst %.loc5_17.4
// CHECK:STDOUT:     %.loc5_17.2: @F.%.loc5_17.6 (@F.%.loc5_17.6) = splice_inst %.loc5_17.5
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_17.3: %i32 = splice_inst %.loc5_17.7
// CHECK:STDOUT:     %n: %i32 = value_binding n, %.loc5_17.3
// CHECK:STDOUT:     %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_10.2(%n.ref)
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%d.param: %D) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %d.ref: %D = name_ref d, %d
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%D) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%d.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%D
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d8d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.37d
// CHECK:STDOUT:   %.loc5_17.4 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.loc5_17.5 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc5_17.6 => <error>
// CHECK:STDOUT:   %.loc5_17.7 => constants.%inst.splice_block
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_member_wrong_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc4 [concrete]
// CHECK:STDOUT:   %F.708: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %E: type = class_type @E [concrete]
// CHECK:STDOUT:   %F.76a: type = class_type @F.loc10 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %E.elem: type = unbound_element_type %E, %F.76a [concrete]
// CHECK:STDOUT:   %struct_type.n.88b: type = struct_type {.n: %F.76a} [concrete]
// CHECK:STDOUT:   %complete_type.15a: <witness> = complete_type_witness %struct_type.n.88b [concrete]
// CHECK:STDOUT:   %pattern_type.849: type = pattern_type %E [concrete]
// CHECK:STDOUT:   %Test.type: type = fn_type @Test [concrete]
// CHECK:STDOUT:   %Test: %Test.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.708, @F.loc4(%E) [concrete]
// CHECK:STDOUT:   %inst.as_compatible: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.660: %E = as_compatible @F.loc4.%x.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.124: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.cdf: %F.76a = splice_block %.6fd {
// CHECK:STDOUT:       %n.ref: %E.elem = name_ref n, @E.%.loc11 [concrete = @E.%.loc11]
// CHECK:STDOUT:       %.cce: ref %F.76a = class_element_access %.660, element0
// CHECK:STDOUT:       %.6fd: %F.76a = acquire_value %.cce
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %inst.splice_block.db7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.cbb: <error> = splice_block <error> [concrete = <error>] {
// CHECK:STDOUT:       %.eb0: %i32 = converted %.cdf, <error> [concrete = <error>]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .E = %E.decl
// CHECK:STDOUT:     .Test = %Test.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F.loc4 [concrete = constants.%F.708] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0, template [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.loc4.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.loc4.%pattern_type (%pattern_type.51d1c4.1) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc4 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc4: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc4_34: Core.Form = init_form %i32.loc4 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.loc4.%T.loc4_16.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_16.2 [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @F.loc4.%T.loc4_16.1 (%T.67db0b.1) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %E.decl: type = class_decl @E [concrete = constants.%E] {} {}
// CHECK:STDOUT:   %Test.decl: %Test.type = fn_decl @Test [concrete = constants.%Test] {
// CHECK:STDOUT:     %e.param_patt: %pattern_type.849 = value_param_pattern [concrete]
// CHECK:STDOUT:     %e.patt: %pattern_type.849 = at_binding_pattern e, %e.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %e.param: %E = value_param call_param0
// CHECK:STDOUT:     %E.ref: type = name_ref E, file.%E.decl [concrete = constants.%E]
// CHECK:STDOUT:     %e: %E = value_binding e, %e.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @E {
// CHECK:STDOUT:   %F.decl: type = class_decl @F.loc10 [concrete = constants.%F.76a] {} {}
// CHECK:STDOUT:   %.loc11: %E.elem = field_decl n, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.n.88b [concrete = constants.%complete_type.15a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%E
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .n = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @F.loc10 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%F.76a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.loc4(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0, template [template = %T.loc4_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc4_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc5_17.4: <instruction> = refine_type_action %x.ref, %T.loc4_16.1 [template]
// CHECK:STDOUT:   %.loc5_17.5: <instruction> = access_member_action %.loc5_17.1, n [template]
// CHECK:STDOUT:   %.loc5_17.6: type = type_of_inst %.loc5_17.5 [template]
// CHECK:STDOUT:   %.loc5_17.7: <instruction> = convert_to_value_action %.loc5_17.2, constants.%i32 [template]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.loc4.%T.loc4_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: %pattern_type.6b6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.ref: @F.loc4.%T.loc4_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc5_17.1: @F.loc4.%T.loc4_16.1 (%T.67db0b.1) = splice_inst %.loc5_17.4
// CHECK:STDOUT:     %.loc5_17.2: @F.loc4.%.loc5_17.6 (@F.loc4.%.loc5_17.6) = splice_inst %.loc5_17.5
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_17.3: %i32 = splice_inst %.loc5_17.7
// CHECK:STDOUT:     %n: %i32 = value_binding n, %.loc5_17.3
// CHECK:STDOUT:     %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %n.ref, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %n.ref, %specific_fn
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_10.2(%n.ref)
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Test(%e.param: %E) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F.708]
// CHECK:STDOUT:   %e.ref: %E = name_ref e, %e
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F.loc4(constants.%E) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%e.ref)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc4(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc4(constants.%E) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%E
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.849
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.15a
// CHECK:STDOUT:   %.loc5_17.4 => constants.%inst.as_compatible
// CHECK:STDOUT:   %.loc5_17.5 => constants.%inst.splice_block.124
// CHECK:STDOUT:   %.loc5_17.6 => constants.%F.76a
// CHECK:STDOUT:   %.loc5_17.7 => constants.%inst.splice_block.db7
// CHECK:STDOUT: }
// CHECK:STDOUT:

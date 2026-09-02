// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/template/convert.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/template/convert.carbon

// --- convert.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[template T: type](x: T) -> i32 {
  let n: i32 = x;
  return n;
}
//@dump-sem-ir-end

fn Test1(n: i32) -> i32 {
  return F(n);
}

class C {
  var n: i32;
  impl as Core.ImplicitAs(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) -> i32 {
  return F(c);
}

// --- fail_cannot_convert.carbon

library "[[@TEST_NAME]]";

fn F[template T: type](x: T) -> i32 {
  let n: i32 = x;
  return n;
}

class D {}

fn Test(d: D) -> i32 {
  // CHECK:STDERR: fail_cannot_convert.carbon:[[@LINE+10]]:10: error: unable to monomorphize specific `F(D)` [ResolvingSpecificHere]
  // CHECK:STDERR:   return F(d);
  // CHECK:STDERR:          ^
  // CHECK:STDERR: fail_cannot_convert.carbon:[[@LINE-10]]:16: note: cannot implicitly convert expression of type `D` to `i32` [ConversionFailure]
  // CHECK:STDERR:   let n: i32 = x;
  // CHECK:STDERR:                ^
  // CHECK:STDERR: fail_cannot_convert.carbon:[[@LINE-13]]:16: note: type `D` does not implement interface `Core.ImplicitAs(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let n: i32 = x;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
  return F(d);
}

// --- as.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn AsI32[template T: type](x: T) -> i32 {
  return x as i32;
}
//@dump-sem-ir-end

fn Test1(n: i32) {
  let _: i32 = AsI32(n);
}

class C {
  var n: i32;
  impl as Core.As(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) {
  let _: i32 = AsI32(c);
}

// --- fail_cannot_as.carbon

library "[[@TEST_NAME]]";

fn AsI32[template T: type](x: T) -> i32 {
  return x as i32;
}

class D {}

fn Test(d: D) {
  // CHECK:STDERR: fail_cannot_as.carbon:[[@LINE+10]]:16: error: unable to monomorphize specific `AsI32(D)` [ResolvingSpecificHere]
  // CHECK:STDERR:   let _: i32 = AsI32(d);
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR: fail_cannot_as.carbon:[[@LINE-9]]:10: note: cannot convert expression of type `D` to `i32` with `as` [ConversionFailure]
  // CHECK:STDERR:   return x as i32;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR: fail_cannot_as.carbon:[[@LINE-12]]:10: note: type `D` does not implement interface `Core.As(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   return x as i32;
  // CHECK:STDERR:          ^~~~~~~~
  // CHECK:STDERR:
  let _: i32 = AsI32(d);
}

// --- as_type.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn AsType[template T: type](x: T, template U: type) {
  let _: U = x as U;
}
//@dump-sem-ir-end

fn Test1(n: i32) {
  AsType(n, i32);
}

class C {
  var n: i32;
  impl as Core.As(i32) {
    fn Convert(self) -> i32 { return self.n; }
  }
}

fn Test2(c: C) {
  AsType(c, i32);
}

// --- fail_cannot_as_type.carbon

library "[[@TEST_NAME]]";

fn AsType[template T: type](x: T, template U: type) {
  let _: U = x as U;
}

class D {}

fn Test(d: D) {
  // CHECK:STDERR: fail_cannot_as_type.carbon:[[@LINE+10]]:3: error: unable to monomorphize specific `AsType(D, i32)` [ResolvingSpecificHere]
  // CHECK:STDERR:   AsType(d, i32);
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR: fail_cannot_as_type.carbon:[[@LINE-9]]:14: note: cannot convert expression of type `D` to `i32` with `as` [ConversionFailure]
  // CHECK:STDERR:   let _: U = x as U;
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR: fail_cannot_as_type.carbon:[[@LINE-12]]:14: note: type `D` does not implement interface `Core.As(i32)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   let _: U = x as U;
  // CHECK:STDERR:              ^~~~~~
  // CHECK:STDERR:
  AsType(d, i32);
}

// CHECK:STDOUT: --- convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47011.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795f: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %n.patt.c48: %pattern_type.6b6 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %.098: %T.67db0b.1 = splice_inst @F.%.loc6_16.3 [template]
// CHECK:STDOUT:   %.c85: %i32 = splice_inst @F.%.loc6_16.5 [template]
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
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.edb: <witness> = impl_witness @C.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type: type = fn_type @C.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert: %C.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value %C, (%ImplicitAs.impl_witness.edb) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.5bd: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.0d5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.3e2: %i32 = specific_inst @F.%x.ref, @F(%i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.e52: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.31c: %i32 = splice_block %.3e2 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.669: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.bfd: %C = specific_inst @F.%x.ref, @F(%C)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.ddb: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.5bd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.cae: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.dbd: %i32 = splice_block %.3ff {
// CHECK:STDOUT:       %impl.elem0.a96: %.ddb = impl_witness_access %ImplicitAs.impl_witness.edb, element0 [concrete = %C.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.bfd, %impl.elem0.a96
// CHECK:STDOUT:       %C.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%.bfd)
// CHECK:STDOUT:       %.3ff: %i32 = value_of_initializer %C.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:     }
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
// CHECK:STDOUT:     %x.param_patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_25.1 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_33: Core.Form = init_form %i32.loc5 [concrete = constants.%.795f]
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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_25.2 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc6_16.3: <instruction> = refine_inst_action %x.ref [template]
// CHECK:STDOUT:   %.loc6_16.4: @F.%T.loc5_16.1 (%T.67db0b.1) = splice_inst %.loc6_16.3 [template = %.loc6_16.4 (constants.%.098)]
// CHECK:STDOUT:   %.loc6_16.5: <instruction> = convert_to_value_action %.loc6_16.1, constants.%i32 [template]
// CHECK:STDOUT:   %.loc6_16.6: %i32 = splice_inst %.loc6_16.5 [template = %.loc6_16.6 (constants.%.c85)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_16.1: @F.%T.loc5_16.1 (%T.67db0b.1) = splice_inst %.loc6_16.3 [template = %.loc6_16.4 (constants.%.098)]
// CHECK:STDOUT:     %.loc6_16.2: %i32 = splice_inst %.loc6_16.5 [template = %.loc6_16.6 (constants.%.c85)]
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %.loc6_16.2
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.e3b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.specific_inst.0d5
// CHECK:STDOUT:   %.loc6_16.4 => invalid
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.e52
// CHECK:STDOUT:   %.loc6_16.6 => invalid
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
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.specific_inst.669
// CHECK:STDOUT:   %.loc6_16.4 => invalid
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.cae
// CHECK:STDOUT:   %.loc6_16.6 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47011.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.a9a: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e1b: %pattern_type.6b6 = return_slot_pattern %return.param_patt.a9a, %i32 [concrete]
// CHECK:STDOUT:   %AsI32.type: type = fn_type @AsI32 [concrete]
// CHECK:STDOUT:   %AsI32: %AsI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %.098: %T.67db0b.1 = splice_inst @AsI32.%.loc6_12.3 [template]
// CHECK:STDOUT:   %.7f3: %i32 = splice_inst @AsI32.%.loc6_12.5 [template]
// CHECK:STDOUT:   %.4f1: %i32 = splice_inst @AsI32.%.loc6_18.2 [template]
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.80e: <witness> = impl_witness @C.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert.type: type = fn_type @C.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert: %C.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value %C, (%As.impl_witness.80e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.087: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
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
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.615: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.2f4: %i32 = specific_inst @AsI32.%x.ref, @AsI32(%i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.1f8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.727: %i32 = splice_block %.2f4 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.a55: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f08: init %i32 = splice_block %Int.as.Copy.impl.Op.call.a7e {
// CHECK:STDOUT:       %impl.elem0.88228e.2: %.737 = impl_witness_access %Copy.impl_witness.b51, element0 [concrete = %Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:       %bound_method.5d3: <bound method> = bound_method %.727, %impl.elem0.88228e.2
// CHECK:STDOUT:       %specific_fn.96a233.2: <specific function> = specific_function %impl.elem0.88228e.2, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.d8a: <bound method> = bound_method %.727, %specific_fn.96a233.2
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call.a7e: init %i32 = call %bound_method.d8a(%.727)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.281: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.71c: %C = specific_inst @AsI32.%x.ref, @AsI32(%C)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %As.WithSelf.Convert.type.087, %As.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.734: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fd9: %i32 = splice_block %.094 {
// CHECK:STDOUT:       %impl.elem0.47b: %.419 = impl_witness_access %As.impl_witness.80e, element0 [concrete = %C.as.As.impl.Convert]
// CHECK:STDOUT:       %bound_method.739: <bound method> = bound_method %.71c, %impl.elem0.47b
// CHECK:STDOUT:       %C.as.As.impl.Convert.call: init %i32 = call %bound_method.739(%.71c)
// CHECK:STDOUT:       %.094: %i32 = value_of_initializer %C.as.As.impl.Convert.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.575: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.297: init %i32 = splice_block %Int.as.Copy.impl.Op.call.03b {
// CHECK:STDOUT:       %impl.elem0.88228e.1: %.737 = impl_witness_access %Copy.impl_witness.b51, element0 [concrete = %Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:       %bound_method.649: <bound method> = bound_method %.fd9, %impl.elem0.88228e.1
// CHECK:STDOUT:       %specific_fn.96a233.1: <specific function> = specific_function %impl.elem0.88228e.1, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.5e5: <bound method> = bound_method %.fd9, %specific_fn.96a233.1
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call.03b: init %i32 = call %bound_method.5e5(%.fd9)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AsI32.decl: %AsI32.type = fn_decl @AsI32 [concrete = constants.%AsI32] {
// CHECK:STDOUT:     %T.patt.loc5_20.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_20.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_29.1: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_29.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_29.1: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_29.1 [template = %x.patt.loc5_29.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_37: Core.Form = init_form %i32.loc5 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_20.2: type = symbolic_binding T, 0, template [template = %T.loc5_20.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @AsI32.%T.loc5_20.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_20.2 [template = %T.loc5_20.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @AsI32.%T.loc5_20.1 (%T.67db0b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AsI32(%T.loc5_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_20.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_20.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:   %T.loc5_20.1: type = symbolic_binding T, 0, template [template = %T.loc5_20.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_20.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc5_29.2: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_29.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_29.2: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_29.2 [template = %x.patt.loc5_29.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_20.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc6_12.3: <instruction> = refine_inst_action %x.ref [template]
// CHECK:STDOUT:   %.loc6_12.4: @AsI32.%T.loc5_20.1 (%T.67db0b.1) = splice_inst %.loc6_12.3 [template = %.loc6_12.4 (constants.%.098)]
// CHECK:STDOUT:   %.loc6_12.5: <instruction> = convert_action %.loc6_12.1, constants.%i32, element7 [template]
// CHECK:STDOUT:   %.loc6_12.6: %i32 = splice_inst %.loc6_12.5 [template = %.loc6_12.6 (constants.%.7f3)]
// CHECK:STDOUT:   %.loc6_18.2: <instruction> = convert_to_category_action %.loc6_12.2, element10 [template]
// CHECK:STDOUT:   %.loc6_18.3: %i32 = splice_inst %.loc6_18.2 [template = %.loc6_18.3 (constants.%.4f1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AsI32.%T.loc5_20.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AsI32.%T.loc5_20.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_12.1: @AsI32.%T.loc5_20.1 (%T.67db0b.1) = splice_inst %.loc6_12.3 [template = %.loc6_12.4 (constants.%.098)]
// CHECK:STDOUT:     %.loc6_12.2: %i32 = splice_inst %.loc6_12.5 [template = %.loc6_12.6 (constants.%.7f3)]
// CHECK:STDOUT:     %.loc6_18.1: %i32 = splice_inst %.loc6_18.2 [template = %.loc6_18.3 (constants.%.4f1)]
// CHECK:STDOUT:     return %.loc6_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsI32(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc5_20.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc5_29.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_29.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsI32(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_20.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_29.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_29.2 => constants.%x.patt.e3b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_12.3 => constants.%inst.specific_inst.615
// CHECK:STDOUT:   %.loc6_12.4 => invalid
// CHECK:STDOUT:   %.loc6_12.5 => constants.%inst.splice_block.1f8
// CHECK:STDOUT:   %.loc6_12.6 => invalid
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.a55
// CHECK:STDOUT:   %.loc6_18.3 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsI32(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc5_20.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc5_29.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc5_29.2 => constants.%x.patt.953
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cdf
// CHECK:STDOUT:   %.loc6_12.3 => constants.%inst.specific_inst.281
// CHECK:STDOUT:   %.loc6_12.4 => invalid
// CHECK:STDOUT:   %.loc6_12.5 => constants.%inst.splice_block.734
// CHECK:STDOUT:   %.loc6_12.6 => invalid
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.575
// CHECK:STDOUT:   %.loc6_18.3 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47011.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %U.patt.014: %pattern_type.98f = symbolic_binding_pattern U, 1, template [template]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1, template [template]
// CHECK:STDOUT:   %AsType.type: type = fn_type @AsType [concrete]
// CHECK:STDOUT:   %AsType: %AsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %U.091 [template]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %U.091 [template]
// CHECK:STDOUT:   %_.patt.972: %pattern_type.946 = value_binding_pattern _ [template]
// CHECK:STDOUT:   %.098: %T.67db0b.1 = splice_inst @AsType.%.loc6_16.4 [template]
// CHECK:STDOUT:   %.51e: %U.091 = splice_inst @AsType.%.loc6_16.6 [template]
// CHECK:STDOUT:   %.f21: %U.091 = splice_inst @AsType.%.loc6_16.8 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %As.type.155: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.80e: <witness> = impl_witness @C.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert.type: type = fn_type @C.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert: %C.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.155 = facet_value %C, (%As.impl_witness.80e) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.087: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cdf: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %_.patt.bed: %pattern_type.6b6 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %inst.specific_inst.fbf: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.5a5: %i32 = specific_inst @AsType.%x.ref, @AsType(%i32, %i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.6e4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f67: %i32 = splice_block %.5a5 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.bce: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.5f4: %i32 = splice_block %.f67 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.specific_inst.842: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c99: %C = specific_inst @AsType.%x.ref, @AsType(%C, %i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %As.WithSelf.Convert.type.087, %As.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.a88: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.781: %i32 = splice_block %.160 {
// CHECK:STDOUT:       %impl.elem0.47b: %.419 = impl_witness_access %As.impl_witness.80e, element0 [concrete = %C.as.As.impl.Convert]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.c99, %impl.elem0.47b
// CHECK:STDOUT:       %C.as.As.impl.Convert.call: init %i32 = call %bound_method(%.c99)
// CHECK:STDOUT:       %.160: %i32 = value_of_initializer %C.as.As.impl.Convert.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.055: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.950: %i32 = splice_block %.781 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AsType.decl: %AsType.type = fn_decl @AsType [concrete = constants.%AsType] {
// CHECK:STDOUT:     %T.patt.loc5_21.1: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_21.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_30.1: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_30.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_30.1: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_30.1 [template = %x.patt.loc5_30.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %U.patt.loc5_45.1: %pattern_type.98f = symbolic_binding_pattern U, 1, template [template = %U.patt.loc5_45.2 (constants.%U.patt.014)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_23.1: type = splice_block %.loc5_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_21: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_21.2: type = symbolic_binding T, 0, template [template = %T.loc5_21.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @AsType.%T.loc5_21.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_21.2 [template = %T.loc5_21.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @AsType.%T.loc5_21.1 (%T.67db0b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc5_47.1: type = splice_block %.loc5_47.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_45: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_47.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_45.2: type = symbolic_binding U, 1, template [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AsType(%T.loc5_21.2: type, %U.loc5_45.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_21.2: %pattern_type.98f = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_21.2 (constants.%T.patt.d47011.1)]
// CHECK:STDOUT:   %T.loc5_21.1: type = symbolic_binding T, 0, template [template = %T.loc5_21.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type.loc5: type = pattern_type %T.loc5_21.1 [template = %pattern_type.loc5 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc5_30.2: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_30.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_30.2: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_30.2 [template = %x.patt.loc5_30.2 (constants.%x.patt.260)]
// CHECK:STDOUT:   %U.patt.loc5_45.2: %pattern_type.98f = symbolic_binding_pattern U, 1, template [template = %U.patt.loc5_45.2 (constants.%U.patt.014)]
// CHECK:STDOUT:   %U.loc5_45.1: type = symbolic_binding U, 1, template [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5: <witness> = require_complete_type %T.loc5_21.1 [template = %require_complete.loc5 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %U.loc5_45.1 [template = %require_complete.loc6 (constants.%require_complete.441)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %U.loc5_45.1 [template = %pattern_type.loc6 (constants.%pattern_type.946)]
// CHECK:STDOUT:   %_.patt.loc6_8.2: @AsType.%pattern_type.loc6 (%pattern_type.946) = value_binding_pattern _ [template = %_.patt.loc6_8.2 (constants.%_.patt.972)]
// CHECK:STDOUT:   %.loc6_16.4: <instruction> = refine_inst_action %x.ref [template]
// CHECK:STDOUT:   %.loc6_16.5: @AsType.%T.loc5_21.1 (%T.67db0b.1) = splice_inst %.loc6_16.4 [template = %.loc6_16.5 (constants.%.098)]
// CHECK:STDOUT:   %.loc6_16.6: <instruction> = convert_action %.loc6_16.1, %U.loc5_45.1, element7 [template]
// CHECK:STDOUT:   %.loc6_16.7: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.6 [template = %.loc6_16.7 (constants.%.51e)]
// CHECK:STDOUT:   %.loc6_16.8: <instruction> = convert_to_category_action %.loc6_16.2, element1 [template]
// CHECK:STDOUT:   %.loc6_16.9: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.8 [template = %.loc6_16.9 (constants.%.f21)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AsType.%T.loc5_21.1 (%T.67db0b.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AsType.%T.loc5_21.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc6_19: type = name_ref U, %U.loc5_45.2 [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:     %.loc6_16.1: @AsType.%T.loc5_21.1 (%T.67db0b.1) = splice_inst %.loc6_16.4 [template = %.loc6_16.5 (constants.%.098)]
// CHECK:STDOUT:     %.loc6_16.2: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.6 [template = %.loc6_16.7 (constants.%.51e)]
// CHECK:STDOUT:     %.loc6_16.3: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.8 [template = %.loc6_16.9 (constants.%.f21)]
// CHECK:STDOUT:     %U.ref.loc6_10: type = name_ref U, %U.loc5_45.2 [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:     %_: @AsType.%U.loc5_45.1 (%U.091) = wrapper_binding _, %.loc6_16.3
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt.loc6_8.1: @AsType.%pattern_type.loc6 (%pattern_type.946) = value_binding_pattern _ [template = %_.patt.loc6_8.2 (constants.%_.patt.972)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsType(constants.%T.67db0b.1, constants.%U.091) {
// CHECK:STDOUT:   %T.patt.loc5_21.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_21.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc5_30.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_30.2 => constants.%x.patt.260
// CHECK:STDOUT:   %U.patt.loc5_45.2 => constants.%U.patt.014
// CHECK:STDOUT:   %U.loc5_45.1 => constants.%U.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsType(constants.%i32, constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_21.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_21.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_30.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_30.2 => constants.%x.patt.e3b
// CHECK:STDOUT:   %U.patt.loc5_45.2 => constants.%U.patt.014
// CHECK:STDOUT:   %U.loc5_45.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %_.patt.loc6_8.2 => constants.%_.patt.bed
// CHECK:STDOUT:   %.loc6_16.4 => constants.%inst.specific_inst.fbf
// CHECK:STDOUT:   %.loc6_16.5 => invalid
// CHECK:STDOUT:   %.loc6_16.6 => constants.%inst.splice_block.6e4
// CHECK:STDOUT:   %.loc6_16.7 => invalid
// CHECK:STDOUT:   %.loc6_16.8 => constants.%inst.splice_block.bce
// CHECK:STDOUT:   %.loc6_16.9 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsType(constants.%C, constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_21.2 => constants.%T.patt.d47011.1
// CHECK:STDOUT:   %T.loc5_21.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc5_30.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc5_30.2 => constants.%x.patt.953
// CHECK:STDOUT:   %U.patt.loc5_45.2 => constants.%U.patt.014
// CHECK:STDOUT:   %U.loc5_45.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5 => constants.%complete_type.cdf
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %_.patt.loc6_8.2 => constants.%_.patt.bed
// CHECK:STDOUT:   %.loc6_16.4 => constants.%inst.specific_inst.842
// CHECK:STDOUT:   %.loc6_16.5 => invalid
// CHECK:STDOUT:   %.loc6_16.6 => constants.%inst.splice_block.a88
// CHECK:STDOUT:   %.loc6_16.7 => invalid
// CHECK:STDOUT:   %.loc6_16.8 => constants.%inst.splice_block.055
// CHECK:STDOUT:   %.loc6_16.9 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

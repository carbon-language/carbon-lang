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

// --- initialize_dependent_target_from_literal.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F(template T: type) {
  var unused w: T = 0;
}
//@dump-sem-ir-end

fn G() {
  F(i32);
}

// --- initialize_primitive.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[template T: type](x: T) {
  var unused w: i32 = x;
}
//@dump-sem-ir-end

fn G() {
  F(0 as i32);
}

// --- initialize_from_tuple.carbon

library "[[@TEST_NAME]]";

class X {}
fn Make() -> X;

//@dump-sem-ir-begin
fn F(template T: type) {
  var unused w: T = (Make(), Make(), Make());
}
//@dump-sem-ir-end

class Y {}

impl (X, X, X) as Core.ImplicitAs(Y) {
  fn Convert(unused self: (X, X, X)) -> Y { return {}; }
}

fn G() {
  F((X, X, X));

  F(array(X, 3));

  F(Y);
}

// --- initialize_from_nested_literal.carbon

library "[[@TEST_NAME]]";

class X {}
fn Make() -> X;

//@dump-sem-ir-begin
fn F(template T: type) {
  var unused w: T = ({.a = Make(), .b = Make()}, Make());
}
//@dump-sem-ir-end

class Y {
  impl X as Core.ImplicitAs(Y) {
    fn Convert(self: X) -> Y;
  }
}

class Z {
  var a: X;
  var b: Y;
}

fn G() {
  F(({.a: X, .b: X}, X));

  F(({.a: Y, .b: Y}, Y));

  F((Z, Y));
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
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
// CHECK:STDOUT:   %.6ec: %i32 = splice_inst @F.%.loc6_16.2 [template]
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
// CHECK:STDOUT:   %inst.specific_inst: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.3e2: %i32 = specific_inst @F.%x.ref, @F(%i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.ddb: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.5bd, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.e67: %i32 = splice_block %.3ff {
// CHECK:STDOUT:       %.bfd: %C = specific_inst @F.%x.ref, @F(%C)
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
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_25.1 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_33: Core.Form = init_form %i32.loc5 [concrete = constants.%.795f]
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
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_25.2 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc6_16.2: <instruction> = convert_to_value_action %x.ref, constants.%i32 [template]
// CHECK:STDOUT:   %.loc6_16.3: %i32 = splice_inst %.loc6_16.2 [template = %.loc6_16.3 (constants.%.6ec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_16.1: %i32 = splice_inst %.loc6_16.2 [template = %.loc6_16.3 (constants.%.6ec)]
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %.loc6_16.1
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
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.e3b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_16.2 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_16.3 => invalid
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cdf
// CHECK:STDOUT:   %.loc6_16.2 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc6_16.3 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- initialize_dependent_target_from_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %w.patt.f6d: %pattern_type.51d1c4.1 = ref_binding_pattern w [template]
// CHECK:STDOUT:   %w.var_patt.7fe: %pattern_type.51d1c4.1 = var_pattern %w.patt.f6d [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.38e: %T = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4e5: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.4a0: %Destroy.WithSelf.SelfDestruct.type.4e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.71d [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc6_3.11 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc6_3.11 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %w.patt.859: %pattern_type.6b6 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.82c: %pattern_type.6b6 = var_pattern %w.patt.859 [concrete]
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
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6e0: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.3c0: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c7a: init %i32 = splice_block %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = %int_0.3c0] {
// CHECK:STDOUT:       %impl.elem0.0e6: %.b7a = impl_witness_access %ImplicitAs.impl_witness.a2a, element0 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:       %bound_method.6ac: <bound method> = bound_method %int_0.5c6, %impl.elem0.0e6 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.0e6, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.9e8: <bound method> = bound_method %int_0.5c6, %specific_fn [concrete = %bound_method.6e0]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.9e8(%int_0.5c6) [concrete = %int_0.3c0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.1) [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %self.param_patt.331: %pattern_type.956 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.319: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt.331 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.705: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.70d: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt.705 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %Destroy.WithSelf.SubobjectDestroy.d01daf.2, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.b41: %Destroy.type = facet_value %i32, (%custom_witness.f8f19d.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.11b: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.b41) [concrete]
// CHECK:STDOUT:   %.bbc: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.11b, %Destroy.facet.b41 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b4c: <bound method> = splice_block %bound_method.521 {
// CHECK:STDOUT:       %.0a7: ref %i32 = specific_inst @F.%w.var, @F(%i32)
// CHECK:STDOUT:       %impl.elem2: %.bbc = impl_witness_access %custom_witness.f8f19d.2, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.2]
// CHECK:STDOUT:       %bound_method.521: <bound method> = bound_method %.0a7, %impl.elem2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %.b4c(%.0a7)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.71d: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.4e5) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.4a0)]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = ref_param_pattern [concrete = constants.%self.param_patt.331]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.319]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32.builtin = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc6_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = ref_param_pattern [concrete = constants.%self.param_patt.331]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.319]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32.builtin = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc6_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %w.patt.loc6_15.2: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_binding_pattern w [template = %w.patt.loc6_15.2 (constants.%w.patt.f6d)]
// CHECK:STDOUT:   %w.var_patt.loc6_3.2: @F.%pattern_type (%pattern_type.51d1c4.1) = var_pattern %w.patt.loc6_15.2 [template = %w.var_patt.loc6_3.2 (constants.%w.var_patt.7fe)]
// CHECK:STDOUT:   %.loc6_3.6: %tuple.type = initialize_action %int_0, %T.loc5_16.1, %w.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc6_3.6, element0 [template]
// CHECK:STDOUT:   %.loc6_3.7: @F.%T.loc5_16.1 (%T) = splice_inst %tuple.elem0 [template = %.loc6_3.7 (constants.%.38e)]
// CHECK:STDOUT:   %.loc6_3.8: <instruction> = compound_member_access_action %w.var, constants.%assoc2 [template]
// CHECK:STDOUT:   %.loc6_3.9: type = type_of_inst %.loc6_3.8 [template = %.loc6_3.9 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc6_3.10: @F.%.loc6_3.9 (%.8cb) = splice_inst %.loc6_3.8 [template = %.loc6_3.10 (constants.%.c23)]
// CHECK:STDOUT:   %.loc6_3.11: <instruction> = call_action (%.loc6_3.3), true [template]
// CHECK:STDOUT:   %.loc6_3.12: type = type_of_inst %.loc6_3.11 [template = %.loc6_3.12 (constants.%.57c)]
// CHECK:STDOUT:   %.loc6_3.13: @F.%.loc6_3.12 (%.57c) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref @F.%T.loc5_16.1 (%T) = var_storage %w.var_patt.loc6_3.1
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc6_3.1: @F.%T.loc5_16.1 (%T) = splice_inst %tuple.elem0 [template = %.loc6_3.7 (constants.%.38e)]
// CHECK:STDOUT:     assign %w.var, %.loc6_3.1
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %w: ref @F.%T.loc5_16.1 (%T) = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt.loc6_15.1: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_binding_pattern w [template = %w.patt.loc6_15.2 (constants.%w.patt.f6d)]
// CHECK:STDOUT:       %w.var_patt.loc6_3.1: @F.%pattern_type (%pattern_type.51d1c4.1) = var_pattern %w.patt.loc6_15.1 [template = %w.var_patt.loc6_3.2 (constants.%w.var_patt.7fe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6_3.2: type = type_of_inst %.loc6_3.8 [template = %.loc6_3.9 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc6_3.3: @F.%.loc6_3.9 (%.8cb) = splice_inst %.loc6_3.8 [template = %.loc6_3.10 (constants.%.c23)]
// CHECK:STDOUT:     %.loc6_3.4: type = type_of_inst %.loc6_3.11 [template = %.loc6_3.12 (constants.%.57c)]
// CHECK:STDOUT:     %.loc6_3.5: @F.%.loc6_3.12 (%.57c) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.e63)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.1(%self.param: ref %i32.builtin) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %w.patt.loc6_15.2 => constants.%w.patt.859
// CHECK:STDOUT:   %w.var_patt.loc6_3.2 => constants.%w.var_patt.82c
// CHECK:STDOUT:   %.loc6_3.6 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_3.7 => constants.%int_0.3c0
// CHECK:STDOUT:   %.loc6_3.8 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_3.9 => <bound method>
// CHECK:STDOUT:   %.loc6_3.10 => invalid
// CHECK:STDOUT:   %.loc6_3.11 => constants.%inst.call
// CHECK:STDOUT:   %.loc6_3.12 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc6_3.13 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- initialize_primitive.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %w.patt: %pattern_type.6b6 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt: %pattern_type.6b6 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   %tuple.type.f7a: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.a11: %i32 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %self.param_patt.331: %pattern_type.956 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.319: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt.331 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc6_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.705: %pattern_type.6b6 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.70d: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt.705 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
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
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.388: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:       %.3e2: %i32 = specific_inst @F.%x.ref, @F(%i32)
// CHECK:STDOUT:       %impl.elem0.882: %.737 = impl_witness_access %Copy.impl_witness.b51, element0 [concrete = %Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:       %bound_method.606: <bound method> = bound_method %.3e2, %impl.elem0.882
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.882, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.65e: <bound method> = bound_method %.3e2, %specific_fn
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.65e(%.3e2)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.242: %tuple.type.f7a = tuple_value (%inst.splice_block) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.8d2 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = ref_param_pattern [concrete = constants.%self.param_patt.331]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.319]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32.builtin = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc6_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = ref_param_pattern [concrete = constants.%self.param_patt.331]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.319]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32.builtin = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc6_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6b6 = ref_param_pattern [concrete = constants.%self.param_patt.705]
// CHECK:STDOUT:     %self.patt: %pattern_type.6b6 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.70d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_25.1 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @F.%T.loc5_16.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @F.%T.loc5_16.1 (%T.67db0b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc5_16.1: type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_25.2: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_25.2 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc6_3.2: %tuple.type.f7a = initialize_action %x.ref, constants.%i32, %w.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc6_3.2, element0 [template]
// CHECK:STDOUT:   %.loc6_3.3: %i32 = splice_inst %tuple.elem0 [template = %.loc6_3.3 (constants.%.a11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref %i32 = var_storage %w.var_patt
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_3.1: %i32 = splice_inst %tuple.elem0 [template = %.loc6_3.3 (constants.%.a11)]
// CHECK:STDOUT:     assign %w.var, %.loc6_3.1
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %w: ref %i32 = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %pattern_type.6b6 = ref_binding_pattern w [concrete = constants.%w.patt]
// CHECK:STDOUT:       %w.var_patt: %pattern_type.6b6 = var_pattern %w.patt [concrete = constants.%w.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %w.var, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%w.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.1(%self.param: ref %i32.builtin) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.2(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.e3b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_3.2 => constants.%tuple.242
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc6_3.3 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- initialize_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %w.patt.f6d: %pattern_type.51d1c4.1 = ref_binding_pattern w [template]
// CHECK:STDOUT:   %w.var_patt.7fe: %pattern_type.51d1c4.1 = var_pattern %w.patt.f6d [template]
// CHECK:STDOUT:   %tuple.type.92e: type = tuple_type (%X, %X, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.f39: type = tuple_type (<instruction>, <instruction>, <instruction>, <instruction>) [concrete]
// CHECK:STDOUT:   %.4e3: %X = splice_inst @F.%tuple.elem1 [template]
// CHECK:STDOUT:   %.c57: %X = splice_inst @F.%tuple.elem2 [template]
// CHECK:STDOUT:   %.ceb: %X = splice_inst @F.%tuple.elem3 [template]
// CHECK:STDOUT:   %.b00: %T = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4e5: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.4a0: %Destroy.WithSelf.SelfDestruct.type.4e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.71d [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.836: type = facet_type <@ImplicitAs, @ImplicitAs(%Y)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @tuple.type.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.eca: type = pattern_type %tuple.type.92e [concrete]
// CHECK:STDOUT:   %pattern_type.7ab: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %tuple.type.as.ImplicitAs.impl.Convert: %tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.836 = facet_value %tuple.type.92e, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.354: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Y, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %X [concrete]
// CHECK:STDOUT:   %complete_type.8ad: <witness> = complete_type_witness %tuple.type.92e [concrete]
// CHECK:STDOUT:   %w.patt.689: %pattern_type.eca = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.e50: %pattern_type.eca = var_pattern %w.patt.689 [concrete]
// CHECK:STDOUT:   %inst.tuple_init: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.61c5: init %tuple.type.92e to %.b1e3d3.2 = tuple_init (@F.%Make.call.loc9_27, @F.%Make.call.loc9_35, @F.%Make.call.loc9_43)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.4d5: ref %X = splice_block %tuple.elem0 {
// CHECK:STDOUT:       %.b1e3d3.2: ref %tuple.type.92e = specific_inst @F.%w.var, @F(%tuple.type.92e)
// CHECK:STDOUT:       %tuple.elem0: ref %X = tuple_access %.b1e3d3.2, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem1: ref %X = tuple_access %.b1e3d3.2, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem2: ref %X = tuple_access %.b1e3d3.2, element2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.1: %tuple.type.f39 = tuple_value (%inst.tuple_init, %inst.splice_block.426619.1, %inst.tuple_access.426619.1, %inst.tuple_access.426619.2) [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.925: %pattern_type.6cb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.617: %pattern_type.6cb = wrapper_binding_pattern self, %self.param_patt.925 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b09: %pattern_type.eca = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6d4: %pattern_type.eca = wrapper_binding_pattern self, %self.param_patt.b09 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.3, %Destroy.WithSelf.SubobjectDestroy.d01daf.3, %Destroy.WithSelf.SelfDestruct.db3fdb.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.51e: %Destroy.type = facet_value %tuple.type.92e, (%custom_witness.f8f19d.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.614: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.51e) [concrete]
// CHECK:STDOUT:   %.9f8: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.614, %Destroy.facet.51e [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.955: <bound method> = splice_block %bound_method.2c0 {
// CHECK:STDOUT:       %.b1e3d3.1: ref %tuple.type.92e = specific_inst @F.%w.var, @F(%tuple.type.92e)
// CHECK:STDOUT:       %impl.elem2.39d: %.9f8 = impl_witness_access %custom_witness.f8f19d.3, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.3]
// CHECK:STDOUT:       %bound_method.2c0: <bound method> = bound_method %.b1e3d3.1, %impl.elem2.39d
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.67d: init %empty_tuple.type = call %.955(%.b1e3d3.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.3e8: <witness> = complete_type_witness %array_type [concrete]
// CHECK:STDOUT:   %pattern_type.92a: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %w.patt.f13: %pattern_type.92a = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.c83: %pattern_type.92a = var_pattern %w.patt.f13 [concrete]
// CHECK:STDOUT:   %int_0.5c61a1.1: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8115.1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc8af.1: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %inst.array_init: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.dc3: init %array_type to %.c5e285.2 = array_init (@F.%Make.call.loc9_27, @F.%Make.call.loc9_35, @F.%Make.call.loc9_43)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c80: ref %X = splice_block %.f75 {
// CHECK:STDOUT:       %.c5e285.2: ref %array_type = specific_inst @F.%w.var, @F(%array_type)
// CHECK:STDOUT:       %int_0.5c61a1.2: Core.IntLiteral = int_value 0 [concrete = %int_0.5c61a1.1]
// CHECK:STDOUT:       %.f75: ref %X = array_index %.c5e285.2, %int_0.5c61a1.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.53e: ref %X = splice_block %.efc {
// CHECK:STDOUT:       %int_1.5b8115.2: Core.IntLiteral = int_value 1 [concrete = %int_1.5b8115.1]
// CHECK:STDOUT:       %.efc: ref %X = array_index %.c5e285.2, %int_1.5b8115.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.1e2: ref %X = splice_block %.d3b {
// CHECK:STDOUT:       %int_2.ecc8af.2: Core.IntLiteral = int_value 2 [concrete = %int_2.ecc8af.1]
// CHECK:STDOUT:       %.d3b: ref %X = array_index %.c5e285.2, %int_2.ecc8af.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.2: %tuple.type.f39 = tuple_value (%inst.array_init, %inst.splice_block.426619.3, %inst.splice_block.426619.4, %inst.splice_block.426619.5) [concrete]
// CHECK:STDOUT:   %self.param_patt.8dc: %pattern_type.92a = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6fb: %pattern_type.92a = wrapper_binding_pattern self, %self.param_patt.8dc [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.4: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.4, %Destroy.WithSelf.SubobjectDestroy.d01daf.4, %Destroy.WithSelf.SelfDestruct.db3fdb.4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.792: %Destroy.type = facet_value %array_type, (%custom_witness.f8f19d.4) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.387: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.792) [concrete]
// CHECK:STDOUT:   %.e83: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.387, %Destroy.facet.792 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.85a: <bound method> = splice_block %bound_method.5ba {
// CHECK:STDOUT:       %.c5e285.1: ref %array_type = specific_inst @F.%w.var, @F(%array_type)
// CHECK:STDOUT:       %impl.elem2.96d: %.e83 = impl_witness_access %custom_witness.f8f19d.4, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.4]
// CHECK:STDOUT:       %bound_method.5ba: <bound method> = bound_method %.c5e285.1, %impl.elem2.96d
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.014: init %empty_tuple.type = call %.85a(%.c5e285.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.patt.22a: %pattern_type.7ab = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.565: %pattern_type.7ab = var_pattern %w.patt.22a [concrete]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.354, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.5fb: init %Y = splice_block %tuple.type.as.ImplicitAs.impl.Convert.call {
// CHECK:STDOUT:       %impl.elem0.bd3: %.e93 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.7c0: <bound method> = bound_method @F.%.loc9_44, %impl.elem0.bd3
// CHECK:STDOUT:       %.cd3ce6.2: ref %Y = specific_inst @F.%w.var, @F(%Y)
// CHECK:STDOUT:       %.71d: ref %X = temporary @F.%.loc9_3.1, @F.%Make.call.loc9_27
// CHECK:STDOUT:       %.173: %X = acquire_value %.71d
// CHECK:STDOUT:       %.61c8: ref %X = temporary @F.%.loc9_3.2, @F.%Make.call.loc9_35
// CHECK:STDOUT:       %.f2a: %X = acquire_value %.61c8
// CHECK:STDOUT:       %.55f: ref %X = temporary @F.%.loc9_3.3, @F.%Make.call.loc9_43
// CHECK:STDOUT:       %.eee: %X = acquire_value %.55f
// CHECK:STDOUT:       %tuple.9aa: %tuple.type.92e = tuple_value (%.173, %.f2a, %.eee)
// CHECK:STDOUT:       %.1f9: %tuple.type.92e = converted @F.%.loc9_44, %tuple.9aa
// CHECK:STDOUT:       %tuple.type.as.ImplicitAs.impl.Convert.call: init %Y to %.cd3ce6.2 = call %bound_method.7c0(%.1f9)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.1: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.2: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.3: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.3: %tuple.type.f39 = tuple_value (%inst.splice_block.426619.7, %inst.temporary_storage.426619.1, %inst.temporary_storage.426619.2, %inst.temporary_storage.426619.3) [concrete]
// CHECK:STDOUT:   %self.param_patt.1fb: %pattern_type.7ab = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.eee: %pattern_type.7ab = wrapper_binding_pattern self, %self.param_patt.1fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc9_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.5: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.5, %Destroy.WithSelf.SubobjectDestroy.d01daf.5, %Destroy.WithSelf.SelfDestruct.db3fdb.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.09f: %Destroy.type = facet_value %Y, (%custom_witness.f8f19d.5) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.575: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.09f) [concrete]
// CHECK:STDOUT:   %.dbd: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.575, %Destroy.facet.09f [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.477: <bound method> = splice_block %bound_method.5af {
// CHECK:STDOUT:       %.cd3ce6.1: ref %Y = specific_inst @F.%w.var, @F(%Y)
// CHECK:STDOUT:       %impl.elem2.5d8: %.dbd = impl_witness_access %custom_witness.f8f19d.5, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.5]
// CHECK:STDOUT:       %bound_method.5af: <bound method> = bound_method %.cd3ce6.1, %impl.elem2.5d8
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.160: init %empty_tuple.type = call %.477(%.cd3ce6.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.71d: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.4e5) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.4a0)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6cb = ref_param_pattern [concrete = constants.%self.param_patt.925]
// CHECK:STDOUT:     %self.patt: %pattern_type.6cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.617]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6cb = ref_param_pattern [concrete = constants.%self.param_patt.925]
// CHECK:STDOUT:     %self.patt: %pattern_type.6cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.617]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.eca = ref_param_pattern [concrete = constants.%self.param_patt.b09]
// CHECK:STDOUT:     %self.patt: %pattern_type.eca = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6d4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.92e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.92e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.eca = ref_param_pattern [concrete = constants.%self.param_patt.b09]
// CHECK:STDOUT:     %self.patt: %pattern_type.eca = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6d4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.92e = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.92e = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.92a = ref_param_pattern [concrete = constants.%self.param_patt.8dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.92a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6fb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %array_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %array_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc9_3.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.92a = ref_param_pattern [concrete = constants.%self.param_patt.8dc]
// CHECK:STDOUT:     %self.patt: %pattern_type.92a = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.6fb]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %array_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %array_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ab = ref_param_pattern [concrete = constants.%self.param_patt.1fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.7ab = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.eee]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Y = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Y = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc9_3.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ab = ref_param_pattern [concrete = constants.%self.param_patt.1fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.7ab = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.eee]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Y = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Y = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_16.2: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_16.1: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc8_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %w.patt.loc9_15.2: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_binding_pattern w [template = %w.patt.loc9_15.2 (constants.%w.patt.f6d)]
// CHECK:STDOUT:   %w.var_patt.loc9_3.2: @F.%pattern_type (%pattern_type.51d1c4.1) = var_pattern %w.patt.loc9_15.2 [template = %w.var_patt.loc9_3.2 (constants.%w.var_patt.7fe)]
// CHECK:STDOUT:   %.loc9_3.9: %tuple.type.f39 = initialize_action %.loc9_44, %T.loc8_16.1, %w.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc9_3.9, element0 [template]
// CHECK:STDOUT:   %tuple.elem1: <instruction> = tuple_access %.loc9_3.9, element1 [template]
// CHECK:STDOUT:   %.loc9_3.10: %X = splice_inst %tuple.elem1 [template = %.loc9_3.10 (constants.%.4e3)]
// CHECK:STDOUT:   %tuple.elem2: <instruction> = tuple_access %.loc9_3.9, element2 [template]
// CHECK:STDOUT:   %.loc9_3.11: %X = splice_inst %tuple.elem2 [template = %.loc9_3.11 (constants.%.c57)]
// CHECK:STDOUT:   %tuple.elem3: <instruction> = tuple_access %.loc9_3.9, element3 [template]
// CHECK:STDOUT:   %.loc9_3.12: %X = splice_inst %tuple.elem3 [template = %.loc9_3.12 (constants.%.ceb)]
// CHECK:STDOUT:   %.loc9_3.13: @F.%T.loc8_16.1 (%T) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.b00)]
// CHECK:STDOUT:   %.loc9_3.14: <instruction> = compound_member_access_action %w.var, constants.%assoc2 [template]
// CHECK:STDOUT:   %.loc9_3.15: type = type_of_inst %.loc9_3.14 [template = %.loc9_3.15 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc9_3.16: @F.%.loc9_3.15 (%.8cb) = splice_inst %.loc9_3.14 [template = %.loc9_3.16 (constants.%.c23)]
// CHECK:STDOUT:   %.loc9_3.17: <instruction> = call_action (%.loc9_3.6), true [template]
// CHECK:STDOUT:   %.loc9_3.18: type = type_of_inst %.loc9_3.17 [template = %.loc9_3.18 (constants.%.57c)]
// CHECK:STDOUT:   %.loc9_3.19: @F.%.loc9_3.18 (%.57c) = splice_inst %.loc9_3.17 [template = %.loc9_3.19 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref @F.%T.loc8_16.1 (%T) = var_storage %w.var_patt.loc9_3.1
// CHECK:STDOUT:     %Make.ref.loc9_22: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.1: %X = splice_inst %tuple.elem1 [template = %.loc9_3.10 (constants.%.4e3)]
// CHECK:STDOUT:     %Make.call.loc9_27: init %X to %.loc9_3.1 = call %Make.ref.loc9_22()
// CHECK:STDOUT:     %Make.ref.loc9_30: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.2: %X = splice_inst %tuple.elem2 [template = %.loc9_3.11 (constants.%.c57)]
// CHECK:STDOUT:     %Make.call.loc9_35: init %X to %.loc9_3.2 = call %Make.ref.loc9_30()
// CHECK:STDOUT:     %Make.ref.loc9_38: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.3: %X = splice_inst %tuple.elem3 [template = %.loc9_3.12 (constants.%.ceb)]
// CHECK:STDOUT:     %Make.call.loc9_43: init %X to %.loc9_3.3 = call %Make.ref.loc9_38()
// CHECK:STDOUT:     %.loc9_44: %tuple.type.92e = tuple_literal (%Make.call.loc9_27, %Make.call.loc9_35, %Make.call.loc9_43)
// CHECK:STDOUT:     %.loc9_3.4: @F.%T.loc8_16.1 (%T) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.b00)]
// CHECK:STDOUT:     assign %w.var, %.loc9_3.4
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_16.2 [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %w: ref @F.%T.loc8_16.1 (%T) = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt.loc9_15.1: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_binding_pattern w [template = %w.patt.loc9_15.2 (constants.%w.patt.f6d)]
// CHECK:STDOUT:       %w.var_patt.loc9_3.1: @F.%pattern_type (%pattern_type.51d1c4.1) = var_pattern %w.patt.loc9_15.1 [template = %w.var_patt.loc9_3.2 (constants.%w.var_patt.7fe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_3.5: type = type_of_inst %.loc9_3.14 [template = %.loc9_3.15 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc9_3.6: @F.%.loc9_3.15 (%.8cb) = splice_inst %.loc9_3.14 [template = %.loc9_3.16 (constants.%.c23)]
// CHECK:STDOUT:     %.loc9_3.7: type = type_of_inst %.loc9_3.17 [template = %.loc9_3.18 (constants.%.57c)]
// CHECK:STDOUT:     %.loc9_3.8: @F.%.loc9_3.18 (%.57c) = splice_inst %.loc9_3.17 [template = %.loc9_3.19 (constants.%.e63)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %tuple.type.92e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %tuple.type.92e) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %tuple.type.92e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.4(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.4(%self.param: ref %array_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.4(%self.param: ref %array_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.5(%self.param: ref %Y) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.5(%self.param: ref %Y) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.5(%self.param: ref %Y) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.type.92e) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%tuple.type.92e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.8ad
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eca
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.689
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.e50
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.1
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.tuple_init
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.tuple_access.426619.1
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.tuple_access.426619.2
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => invalid
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.1
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%array_type) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%array_type
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3e8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.92a
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.f13
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.c83
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.2
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.array_init
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => invalid
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.2
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Y) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%Y
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ab
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.22a
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.565
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.3
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.7
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.temporary_storage.426619.1
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.temporary_storage.426619.2
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.temporary_storage.426619.3
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.8
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => invalid
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.3
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- initialize_from_nested_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %pattern_type.6cb: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [template]
// CHECK:STDOUT:   %w.patt.f6d: %pattern_type.51d1c4.1 = ref_binding_pattern w [template]
// CHECK:STDOUT:   %w.var_patt.7fe: %pattern_type.51d1c4.1 = var_pattern %w.patt.f6d [template]
// CHECK:STDOUT:   %struct_type.a.b.f38: type = struct_type {.a: %X, .b: %X} [concrete]
// CHECK:STDOUT:   %tuple.type.4c2: type = tuple_type (%struct_type.a.b.f38, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.f39: type = tuple_type (<instruction>, <instruction>, <instruction>, <instruction>) [concrete]
// CHECK:STDOUT:   %.4e3: %X = splice_inst @F.%tuple.elem1 [template]
// CHECK:STDOUT:   %.c57: %X = splice_inst @F.%tuple.elem2 [template]
// CHECK:STDOUT:   %.ceb: %X = splice_inst @F.%tuple.elem3 [template]
// CHECK:STDOUT:   %.b00: %T = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4e5: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.4a0: %Destroy.WithSelf.SelfDestruct.type.4e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.71d [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.836: type = facet_type <@ImplicitAs, @ImplicitAs(%Y)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @X.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ab: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.type: type = fn_type @X.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert: %X.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.836 = facet_value %X, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.bb3: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Y, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Z: type = class_type @Z [concrete]
// CHECK:STDOUT:   %Z.elem.d23: type = unbound_element_type %Z, %X [concrete]
// CHECK:STDOUT:   %Z.elem.5d0: type = unbound_element_type %Z, %Y [concrete]
// CHECK:STDOUT:   %struct_type.a.b.297: type = struct_type {.a: %X, .b: %Y} [concrete]
// CHECK:STDOUT:   %struct_type.a.b.095: type = struct_type {.a: %Y, .b: %Y} [concrete]
// CHECK:STDOUT:   %tuple.type.2f4: type = tuple_type (%struct_type.a.b.095, %Y) [concrete]
// CHECK:STDOUT:   %tuple.type.bb4: type = tuple_type (%Z, %Y) [concrete]
// CHECK:STDOUT:   %complete_type.9dc: <witness> = complete_type_witness %tuple.type.4c2 [concrete]
// CHECK:STDOUT:   %pattern_type.9a0: type = pattern_type %tuple.type.4c2 [concrete]
// CHECK:STDOUT:   %w.patt.a80: %pattern_type.9a0 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.607: %pattern_type.9a0 = var_pattern %w.patt.a80 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.e45: init %tuple.type.4c2 = splice_block %.b73 {
// CHECK:STDOUT:       %.484: init %struct_type.a.b.f38 to %tuple.elem0.a28 = struct_init (@F.%Make.call.loc9_33, @F.%Make.call.loc9_46)
// CHECK:STDOUT:       %.a09: init %struct_type.a.b.f38 = converted @F.%.loc9_47, %.484
// CHECK:STDOUT:       %.b73: init %tuple.type.4c2 to %.250153.2 = tuple_init (%.a09, @F.%Make.call.loc9_55)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.550: ref %X = splice_block %.d62 {
// CHECK:STDOUT:       %.250153.2: ref %tuple.type.4c2 = specific_inst @F.%w.var, @F(%tuple.type.4c2)
// CHECK:STDOUT:       %tuple.elem0.a28: ref %struct_type.a.b.f38 = tuple_access %.250153.2, element0
// CHECK:STDOUT:       %.d62: ref %X = struct_access %tuple.elem0.a28, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.struct_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.162: ref %X = struct_access %tuple.elem0.a28, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem1.79b: ref %X = tuple_access %.250153.2, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.1: %tuple.type.f39 = tuple_value (%inst.splice_block.426619.1, %inst.splice_block.426619.2, %inst.struct_access, %inst.tuple_access) [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_3.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.925: %pattern_type.6cb = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.617: %pattern_type.6cb = wrapper_binding_pattern self, %self.param_patt.925 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.4c3: type = pattern_type %struct_type.a.b.f38 [concrete]
// CHECK:STDOUT:   %self.param_patt.bd8: %pattern_type.4c3 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.7ce: %pattern_type.4c3 = wrapper_binding_pattern self, %self.param_patt.bd8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.165: %pattern_type.9a0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c78: %pattern_type.9a0 = wrapper_binding_pattern self, %self.param_patt.165 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.4: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.4, %Destroy.WithSelf.SubobjectDestroy.d01daf.4, %Destroy.WithSelf.SelfDestruct.db3fdb.4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e65: %Destroy.type = facet_value %tuple.type.4c2, (%custom_witness.f8f19d.4) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.a0f: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.e65) [concrete]
// CHECK:STDOUT:   %.f15: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.a0f, %Destroy.facet.e65 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b25: <bound method> = splice_block %bound_method.773 {
// CHECK:STDOUT:       %.250153.1: ref %tuple.type.4c2 = specific_inst @F.%w.var, @F(%tuple.type.4c2)
// CHECK:STDOUT:       %impl.elem2.69b: %.f15 = impl_witness_access %custom_witness.f8f19d.4, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.4]
// CHECK:STDOUT:       %bound_method.773: <bound method> = bound_method %.250153.1, %impl.elem2.69b
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.074: init %empty_tuple.type = call %.b25(%.250153.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.8ea: <witness> = complete_type_witness %tuple.type.2f4 [concrete]
// CHECK:STDOUT:   %pattern_type.2cd: type = pattern_type %tuple.type.2f4 [concrete]
// CHECK:STDOUT:   %w.patt.e8c: %pattern_type.2cd = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.61a: %pattern_type.2cd = var_pattern %w.patt.e8c [concrete]
// CHECK:STDOUT:   %.fd6: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.bb3, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d1a: init %tuple.type.2f4 = splice_block %.0f6 {
// CHECK:STDOUT:       %impl.elem0.94756c.3: %.fd6 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.d13: <bound method> = bound_method @F.%Make.call.loc9_33, %impl.elem0.94756c.3
// CHECK:STDOUT:       %.9ba: ref %Y = splice_block %.e68 {
// CHECK:STDOUT:         %.fe34fc.2: ref %tuple.type.2f4 = specific_inst @F.%w.var, @F(%tuple.type.2f4)
// CHECK:STDOUT:         %tuple.elem0.25a: ref %struct_type.a.b.095 = tuple_access %.fe34fc.2, element0
// CHECK:STDOUT:         %.e68: ref %Y = struct_access %tuple.elem0.25a, element0
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.71d: ref %X = temporary @F.%.loc9_3.1, @F.%Make.call.loc9_33
// CHECK:STDOUT:       %.173: %X = acquire_value %.71d
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.5f0: init %Y to %.9ba = call %bound_method.d13(%.173)
// CHECK:STDOUT:       %.5ba: init %Y = converted @F.%Make.call.loc9_33, %X.as.ImplicitAs.impl.Convert.call.5f0
// CHECK:STDOUT:       %impl.elem0.94756c.4: %.fd6 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.a267a9.2: <bound method> = bound_method @F.%Make.call.loc9_46, %impl.elem0.94756c.4
// CHECK:STDOUT:       %.9f2: ref %Y = struct_access %tuple.elem0.25a, element1
// CHECK:STDOUT:       %.61c809.2: ref %X = temporary @F.%.loc9_3.2, @F.%Make.call.loc9_46
// CHECK:STDOUT:       %.f2a65d.2: %X = acquire_value %.61c809.2
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.2aa: init %Y to %.9f2 = call %bound_method.a267a9.2(%.f2a65d.2)
// CHECK:STDOUT:       %.eb7: init %Y = converted @F.%Make.call.loc9_46, %X.as.ImplicitAs.impl.Convert.call.2aa
// CHECK:STDOUT:       %.230: init %struct_type.a.b.095 to %tuple.elem0.25a = struct_init (%.5ba, %.eb7)
// CHECK:STDOUT:       %.b92: init %struct_type.a.b.095 = converted @F.%.loc9_47, %.230
// CHECK:STDOUT:       %impl.elem0.94756c.5: %.fd6 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.1fb3b5.2: <bound method> = bound_method @F.%Make.call.loc9_55, %impl.elem0.94756c.5
// CHECK:STDOUT:       %tuple.elem1.0e1: ref %Y = tuple_access %.fe34fc.2, element1
// CHECK:STDOUT:       %.55fb1a.2: ref %X = temporary @F.%.loc9_3.3, @F.%Make.call.loc9_55
// CHECK:STDOUT:       %.eeedf6.2: %X = acquire_value %.55fb1a.2
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.a49: init %Y to %tuple.elem1.0e1 = call %bound_method.1fb3b5.2(%.eeedf6.2)
// CHECK:STDOUT:       %.b3a: init %Y = converted @F.%Make.call.loc9_55, %X.as.ImplicitAs.impl.Convert.call.a49
// CHECK:STDOUT:       %.0f6: init %tuple.type.2f4 to %.fe34fc.2 = tuple_init (%.b92, %.b3a)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.1: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.2: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.3: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.2: %tuple.type.f39 = tuple_value (%inst.splice_block.426619.4, %inst.temporary_storage.426619.1, %inst.temporary_storage.426619.2, %inst.temporary_storage.426619.3) [concrete]
// CHECK:STDOUT:   %self.param_patt.1fb: %pattern_type.7ab = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.eee: %pattern_type.7ab = wrapper_binding_pattern self, %self.param_patt.1fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.5: type = fn_type @Destroy.WithSelf.Op.loc9_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.5: %Destroy.WithSelf.Op.type.ef016f.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.5: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.5: %Destroy.WithSelf.SelfDestruct.type.fbceb5.5 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.83c: type = pattern_type %struct_type.a.b.095 [concrete]
// CHECK:STDOUT:   %self.param_patt.201: %pattern_type.83c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.2b2: %pattern_type.83c = wrapper_binding_pattern self, %self.param_patt.201 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.6: type = fn_type @Destroy.WithSelf.Op.loc9_3.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.6: %Destroy.WithSelf.Op.type.ef016f.6 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.7ad: %pattern_type.2cd = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.be8: %pattern_type.2cd = wrapper_binding_pattern self, %self.param_patt.7ad [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.7: type = fn_type @Destroy.WithSelf.Op.loc9_3.7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.7: %Destroy.WithSelf.Op.type.ef016f.7 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.7: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.7 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.7: %Destroy.WithSelf.SelfDestruct.type.fbceb5.7 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.7: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.7, %Destroy.WithSelf.SubobjectDestroy.d01daf.7, %Destroy.WithSelf.SelfDestruct.db3fdb.7), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.585: %Destroy.type = facet_value %tuple.type.2f4, (%custom_witness.f8f19d.7) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.9d1: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.585) [concrete]
// CHECK:STDOUT:   %.005: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.9d1, %Destroy.facet.585 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.52d: <bound method> = splice_block %bound_method.05a {
// CHECK:STDOUT:       %.fe34fc.1: ref %tuple.type.2f4 = specific_inst @F.%w.var, @F(%tuple.type.2f4)
// CHECK:STDOUT:       %impl.elem2.7f4: %.005 = impl_witness_access %custom_witness.f8f19d.7, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.7]
// CHECK:STDOUT:       %bound_method.05a: <bound method> = bound_method %.fe34fc.1, %impl.elem2.7f4
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.91b: init %empty_tuple.type = call %.52d(%.fe34fc.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.561: <witness> = complete_type_witness %tuple.type.bb4 [concrete]
// CHECK:STDOUT:   %pattern_type.8b0: type = pattern_type %tuple.type.bb4 [concrete]
// CHECK:STDOUT:   %w.patt.d74: %pattern_type.8b0 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.e1e: %pattern_type.8b0 = var_pattern %w.patt.d74 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.20b: init %tuple.type.bb4 = splice_block %.725 {
// CHECK:STDOUT:       %impl.elem0.94756c.1: %.fd6 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.a267a9.1: <bound method> = bound_method @F.%Make.call.loc9_46, %impl.elem0.94756c.1
// CHECK:STDOUT:       %.25e: ref %Y = class_element_access %tuple.elem0.d0d, element1
// CHECK:STDOUT:       %.61c809.1: ref %X = temporary @F.%.loc9_3.2, @F.%Make.call.loc9_46
// CHECK:STDOUT:       %.f2a65d.1: %X = acquire_value %.61c809.1
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.bb4: init %Y to %.25e = call %bound_method.a267a9.1(%.f2a65d.1)
// CHECK:STDOUT:       %.684: init %Y = converted @F.%Make.call.loc9_46, %X.as.ImplicitAs.impl.Convert.call.bb4
// CHECK:STDOUT:       %.175: init %Z to %tuple.elem0.d0d = class_init (@F.%Make.call.loc9_33, %.684)
// CHECK:STDOUT:       %.cca: init %Z = converted @F.%.loc9_47, %.175
// CHECK:STDOUT:       %impl.elem0.94756c.2: %.fd6 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.1fb3b5.1: <bound method> = bound_method @F.%Make.call.loc9_55, %impl.elem0.94756c.2
// CHECK:STDOUT:       %tuple.elem1.c76: ref %Y = tuple_access %.6b01cf.2, element1
// CHECK:STDOUT:       %.55fb1a.1: ref %X = temporary @F.%.loc9_3.3, @F.%Make.call.loc9_55
// CHECK:STDOUT:       %.eeedf6.1: %X = acquire_value %.55fb1a.1
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.19e: init %Y to %tuple.elem1.c76 = call %bound_method.1fb3b5.1(%.eeedf6.1)
// CHECK:STDOUT:       %.b5d: init %Y = converted @F.%Make.call.loc9_55, %X.as.ImplicitAs.impl.Convert.call.19e
// CHECK:STDOUT:       %.725: init %tuple.type.bb4 to %.6b01cf.2 = tuple_init (%.cca, %.b5d)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.1b8: ref %X = splice_block %.b76 {
// CHECK:STDOUT:       %.6b01cf.2: ref %tuple.type.bb4 = specific_inst @F.%w.var, @F(%tuple.type.bb4)
// CHECK:STDOUT:       %tuple.elem0.d0d: ref %Z = tuple_access %.6b01cf.2, element0
// CHECK:STDOUT:       %.b76: ref %X = class_element_access %tuple.elem0.d0d, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.4: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.temporary_storage.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7a56fd.5: ref %X = temporary_storage
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.3: %tuple.type.f39 = tuple_value (%inst.splice_block.426619.6, %inst.splice_block.426619.7, %inst.temporary_storage.426619.4, %inst.temporary_storage.426619.5) [concrete]
// CHECK:STDOUT:   %pattern_type.483: type = pattern_type %struct_type.a.b.297 [concrete]
// CHECK:STDOUT:   %self.param_patt.502: %pattern_type.483 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.93b: %pattern_type.483 = wrapper_binding_pattern self, %self.param_patt.502 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.8: type = fn_type @Destroy.WithSelf.Op.loc9_3.8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.8: %Destroy.WithSelf.Op.type.ef016f.8 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.34e: type = pattern_type %Z [concrete]
// CHECK:STDOUT:   %self.param_patt.37e: %pattern_type.34e = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.f3f: %pattern_type.34e = wrapper_binding_pattern self, %self.param_patt.37e [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.9: type = fn_type @Destroy.WithSelf.Op.loc9_3.9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.9: %Destroy.WithSelf.Op.type.ef016f.9 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.328: %pattern_type.8b0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.289: %pattern_type.8b0 = wrapper_binding_pattern self, %self.param_patt.328 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.10: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.10: type = fn_type @Destroy.WithSelf.Op.loc9_3.10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.10: %Destroy.WithSelf.Op.type.ef016f.10 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.10: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.10 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.10: %Destroy.WithSelf.SelfDestruct.type.fbceb5.10 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.10: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.10, %Destroy.WithSelf.SubobjectDestroy.d01daf.10, %Destroy.WithSelf.SelfDestruct.db3fdb.10), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.128: %Destroy.type = facet_value %tuple.type.bb4, (%custom_witness.f8f19d.10) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.9b9: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.128) [concrete]
// CHECK:STDOUT:   %.e5d: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.9b9, %Destroy.facet.128 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7bc: <bound method> = splice_block %bound_method.782 {
// CHECK:STDOUT:       %.6b01cf.1: ref %tuple.type.bb4 = specific_inst @F.%w.var, @F(%tuple.type.bb4)
// CHECK:STDOUT:       %impl.elem2.1f7: %.e5d = impl_witness_access %custom_witness.f8f19d.10, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.10]
// CHECK:STDOUT:       %bound_method.782: <bound method> = bound_method %.6b01cf.1, %impl.elem2.1f7
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.509: init %empty_tuple.type = call %.7bc(%.6b01cf.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.71d: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.4e5) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.4a0)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_3.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6cb = ref_param_pattern [concrete = constants.%self.param_patt.925]
// CHECK:STDOUT:     %self.patt: %pattern_type.6cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.617]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_3.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.6cb = ref_param_pattern [concrete = constants.%self.param_patt.925]
// CHECK:STDOUT:     %self.patt: %pattern_type.6cb = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.617]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %X = ref_param call_param0
// CHECK:STDOUT:     %self: ref %X = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4c3 = ref_param_pattern [concrete = constants.%self.param_patt.bd8]
// CHECK:STDOUT:     %self.patt: %pattern_type.4c3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.f38 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.f38 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc9_3.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.4c3 = ref_param_pattern [concrete = constants.%self.param_patt.bd8]
// CHECK:STDOUT:     %self.patt: %pattern_type.4c3 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.7ce]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.f38 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.f38 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9a0 = ref_param_pattern [concrete = constants.%self.param_patt.165]
// CHECK:STDOUT:     %self.patt: %pattern_type.9a0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c78]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.4c2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.4c2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc9_3.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9a0 = ref_param_pattern [concrete = constants.%self.param_patt.165]
// CHECK:STDOUT:     %self.patt: %pattern_type.9a0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c78]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.4c2 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.4c2 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.5: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.5 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.5 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ab = ref_param_pattern [concrete = constants.%self.param_patt.1fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.7ab = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.eee]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Y = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Y = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.5: %Destroy.WithSelf.Op.type.ef016f.5 = fn_decl @Destroy.WithSelf.Op.loc9_3.5 [concrete = constants.%Destroy.WithSelf.Op.403171.5] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7ab = ref_param_pattern [concrete = constants.%self.param_patt.1fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.7ab = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.eee]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Y = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Y = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.6: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.6 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.83c = ref_param_pattern [concrete = constants.%self.param_patt.201]
// CHECK:STDOUT:     %self.patt: %pattern_type.83c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.2b2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.095 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.095 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.6: %Destroy.WithSelf.Op.type.ef016f.6 = fn_decl @Destroy.WithSelf.Op.loc9_3.6 [concrete = constants.%Destroy.WithSelf.Op.403171.6] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.83c = ref_param_pattern [concrete = constants.%self.param_patt.201]
// CHECK:STDOUT:     %self.patt: %pattern_type.83c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.2b2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.095 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.095 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.7: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.7 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.7 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2cd = ref_param_pattern [concrete = constants.%self.param_patt.7ad]
// CHECK:STDOUT:     %self.patt: %pattern_type.2cd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.be8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.2f4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.2f4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.7: %Destroy.WithSelf.Op.type.ef016f.7 = fn_decl @Destroy.WithSelf.Op.loc9_3.7 [concrete = constants.%Destroy.WithSelf.Op.403171.7] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.2cd = ref_param_pattern [concrete = constants.%self.param_patt.7ad]
// CHECK:STDOUT:     %self.patt: %pattern_type.2cd = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.be8]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.2f4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.2f4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.8: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.8 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.8 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.483 = ref_param_pattern [concrete = constants.%self.param_patt.502]
// CHECK:STDOUT:     %self.patt: %pattern_type.483 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.93b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.297 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.297 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.8: %Destroy.WithSelf.Op.type.ef016f.8 = fn_decl @Destroy.WithSelf.Op.loc9_3.8 [concrete = constants.%Destroy.WithSelf.Op.403171.8] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.483 = ref_param_pattern [concrete = constants.%self.param_patt.502]
// CHECK:STDOUT:     %self.patt: %pattern_type.483 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.93b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.a.b.297 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.a.b.297 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.9: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.9 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.9 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.34e = ref_param_pattern [concrete = constants.%self.param_patt.37e]
// CHECK:STDOUT:     %self.patt: %pattern_type.34e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f3f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Z = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Z = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.9: %Destroy.WithSelf.Op.type.ef016f.9 = fn_decl @Destroy.WithSelf.Op.loc9_3.9 [concrete = constants.%Destroy.WithSelf.Op.403171.9] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.34e = ref_param_pattern [concrete = constants.%self.param_patt.37e]
// CHECK:STDOUT:     %self.patt: %pattern_type.34e = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.f3f]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Z = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Z = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.10: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.10 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_3.10 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.10] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8b0 = ref_param_pattern [concrete = constants.%self.param_patt.328]
// CHECK:STDOUT:     %self.patt: %pattern_type.8b0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.289]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.bb4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.bb4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.10: %Destroy.WithSelf.Op.type.ef016f.10 = fn_decl @Destroy.WithSelf.Op.loc9_3.10 [concrete = constants.%Destroy.WithSelf.Op.403171.10] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8b0 = ref_param_pattern [concrete = constants.%self.param_patt.328]
// CHECK:STDOUT:     %self.patt: %pattern_type.8b0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.289]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %tuple.type.bb4 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %tuple.type.bb4 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_16.2: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_16.1: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc8_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %w.patt.loc9_15.2: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_binding_pattern w [template = %w.patt.loc9_15.2 (constants.%w.patt.f6d)]
// CHECK:STDOUT:   %w.var_patt.loc9_3.2: @F.%pattern_type (%pattern_type.51d1c4.1) = var_pattern %w.patt.loc9_15.2 [template = %w.var_patt.loc9_3.2 (constants.%w.var_patt.7fe)]
// CHECK:STDOUT:   %.loc9_3.9: %tuple.type.f39 = initialize_action %.loc9_56, %T.loc8_16.1, %w.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc9_3.9, element0 [template]
// CHECK:STDOUT:   %tuple.elem1: <instruction> = tuple_access %.loc9_3.9, element1 [template]
// CHECK:STDOUT:   %.loc9_3.10: %X = splice_inst %tuple.elem1 [template = %.loc9_3.10 (constants.%.4e3)]
// CHECK:STDOUT:   %tuple.elem2: <instruction> = tuple_access %.loc9_3.9, element2 [template]
// CHECK:STDOUT:   %.loc9_3.11: %X = splice_inst %tuple.elem2 [template = %.loc9_3.11 (constants.%.c57)]
// CHECK:STDOUT:   %tuple.elem3: <instruction> = tuple_access %.loc9_3.9, element3 [template]
// CHECK:STDOUT:   %.loc9_3.12: %X = splice_inst %tuple.elem3 [template = %.loc9_3.12 (constants.%.ceb)]
// CHECK:STDOUT:   %.loc9_3.13: @F.%T.loc8_16.1 (%T) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.b00)]
// CHECK:STDOUT:   %.loc9_3.14: <instruction> = compound_member_access_action %w.var, constants.%assoc2 [template]
// CHECK:STDOUT:   %.loc9_3.15: type = type_of_inst %.loc9_3.14 [template = %.loc9_3.15 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc9_3.16: @F.%.loc9_3.15 (%.8cb) = splice_inst %.loc9_3.14 [template = %.loc9_3.16 (constants.%.c23)]
// CHECK:STDOUT:   %.loc9_3.17: <instruction> = call_action (%.loc9_3.6), true [template]
// CHECK:STDOUT:   %.loc9_3.18: type = type_of_inst %.loc9_3.17 [template = %.loc9_3.18 (constants.%.57c)]
// CHECK:STDOUT:   %.loc9_3.19: @F.%.loc9_3.18 (%.57c) = splice_inst %.loc9_3.17 [template = %.loc9_3.19 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref @F.%T.loc8_16.1 (%T) = var_storage %w.var_patt.loc9_3.1
// CHECK:STDOUT:     %Make.ref.loc9_28: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.1: %X = splice_inst %tuple.elem1 [template = %.loc9_3.10 (constants.%.4e3)]
// CHECK:STDOUT:     %Make.call.loc9_33: init %X to %.loc9_3.1 = call %Make.ref.loc9_28()
// CHECK:STDOUT:     %Make.ref.loc9_41: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.2: %X = splice_inst %tuple.elem2 [template = %.loc9_3.11 (constants.%.c57)]
// CHECK:STDOUT:     %Make.call.loc9_46: init %X to %.loc9_3.2 = call %Make.ref.loc9_41()
// CHECK:STDOUT:     %.loc9_47: %struct_type.a.b.f38 = struct_literal (%Make.call.loc9_33, %Make.call.loc9_46)
// CHECK:STDOUT:     %Make.ref.loc9_50: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.3: %X = splice_inst %tuple.elem3 [template = %.loc9_3.12 (constants.%.ceb)]
// CHECK:STDOUT:     %Make.call.loc9_55: init %X to %.loc9_3.3 = call %Make.ref.loc9_50()
// CHECK:STDOUT:     %.loc9_56: %tuple.type.4c2 = tuple_literal (%.loc9_47, %Make.call.loc9_55)
// CHECK:STDOUT:     %.loc9_3.4: @F.%T.loc8_16.1 (%T) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.b00)]
// CHECK:STDOUT:     assign %w.var, %.loc9_3.4
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_16.2 [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:     %w: ref @F.%T.loc8_16.1 (%T) = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt.loc9_15.1: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_binding_pattern w [template = %w.patt.loc9_15.2 (constants.%w.patt.f6d)]
// CHECK:STDOUT:       %w.var_patt.loc9_3.1: @F.%pattern_type (%pattern_type.51d1c4.1) = var_pattern %w.patt.loc9_15.1 [template = %w.var_patt.loc9_3.2 (constants.%w.var_patt.7fe)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_3.5: type = type_of_inst %.loc9_3.14 [template = %.loc9_3.15 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc9_3.6: @F.%.loc9_3.15 (%.8cb) = splice_inst %.loc9_3.14 [template = %.loc9_3.16 (constants.%.c23)]
// CHECK:STDOUT:     %.loc9_3.7: type = type_of_inst %.loc9_3.17 [template = %.loc9_3.18 (constants.%.57c)]
// CHECK:STDOUT:     %.loc9_3.8: @F.%.loc9_3.18 (%.57c) = splice_inst %.loc9_3.17 [template = %.loc9_3.19 (constants.%.e63)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.2(%self.param: ref %X) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.2(%self.param: ref %X) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.3(%self.param: ref %struct_type.a.b.f38) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_3.1: ref %X = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %X = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.3(%self.param: ref %struct_type.a.b.f38) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.3(%self.param: ref %struct_type.a.b.f38) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.4(%self.param: ref %tuple.type.4c2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.4(%self.param: ref %tuple.type.4c2) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.4(%self.param: ref %tuple.type.4c2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.5(%self.param: ref %Y) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.5(%self.param: ref %Y) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.5(%self.param: ref %Y) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.5(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.5(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.6(%self.param: ref %struct_type.a.b.095) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_3.1: ref %Y = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %Y = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.6(%self.param: ref %struct_type.a.b.095) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.6(%self.param: ref %struct_type.a.b.095) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.6(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.6(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.7(%self.param: ref %tuple.type.2f4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.7(%self.param: ref %tuple.type.2f4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.7(%self.param: ref %tuple.type.2f4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.7(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.7(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.8(%self.param: ref %struct_type.a.b.297) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_3.1: ref %X = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %Y = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.8(%self.param: ref %struct_type.a.b.297) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.8(%self.param: ref %struct_type.a.b.297) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.8(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.8(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.9(%self.param: ref %Z) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: %Z.elem.d23 = name_ref a, @Z.%field_decl.loc20 [concrete = @Z.%field_decl.loc20]
// CHECK:STDOUT:   %.loc9_3.1: ref %X = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %b.ref: %Z.elem.5d0 = name_ref b, @Z.%field_decl.loc21 [concrete = @Z.%field_decl.loc21]
// CHECK:STDOUT:   %.loc9_3.2: ref %Y = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.9(%self.param: ref %Z) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.9(%self.param: ref %Z) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.9(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.9(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.10(%self.param: ref %tuple.type.bb4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.10(%self.param: ref %tuple.type.bb4) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.10(%self.param: ref %tuple.type.bb4) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.10(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.10(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.type.4c2) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%tuple.type.4c2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.9dc
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9a0
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.a80
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.607
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.1
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.struct_access
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.tuple_access
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => invalid
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.1
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.type.2f4) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%tuple.type.2f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.8ea
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2cd
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.e8c
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.61a
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.2
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.temporary_storage.426619.1
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.temporary_storage.426619.2
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.temporary_storage.426619.3
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => invalid
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.2
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.type.bb4) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%tuple.type.bb4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.561
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8b0
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.d74
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.e1e
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.3
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.splice_block.426619.7
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.temporary_storage.426619.4
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.temporary_storage.426619.5
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.8
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => invalid
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.3
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
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
// CHECK:STDOUT:   %.865: %i32 = splice_inst @AsI32.%.loc6_12.2 [template]
// CHECK:STDOUT:   %.c28: %i32 = splice_inst @AsI32.%.loc6_18.2 [template]
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
// CHECK:STDOUT:   %inst.specific_inst: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.2f4: %i32 = specific_inst @AsI32.%x.ref, @AsI32(%i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7c4: init %i32 = splice_block %Int.as.Copy.impl.Op.call.556 {
// CHECK:STDOUT:       %impl.elem0.88228e.2: %.737 = impl_witness_access %Copy.impl_witness.b51, element0 [concrete = %Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:       %bound_method.4e3: <bound method> = bound_method %.2f4, %impl.elem0.88228e.2
// CHECK:STDOUT:       %specific_fn.96a233.2: <specific function> = specific_function %impl.elem0.88228e.2, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.7bf: <bound method> = bound_method %.2f4, %specific_fn.96a233.2
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call.556: init %i32 = call %bound_method.7bf(%.2f4)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %As.WithSelf.Convert.type.087, %As.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c1e: %i32 = splice_block %.094 {
// CHECK:STDOUT:       %.71c: %C = specific_inst @AsI32.%x.ref, @AsI32(%C)
// CHECK:STDOUT:       %impl.elem0.47b: %.419 = impl_witness_access %As.impl_witness.80e, element0 [concrete = %C.as.As.impl.Convert]
// CHECK:STDOUT:       %bound_method.739: <bound method> = bound_method %.71c, %impl.elem0.47b
// CHECK:STDOUT:       %C.as.As.impl.Convert.call: init %i32 = call %bound_method.739(%.71c)
// CHECK:STDOUT:       %.094: %i32 = value_of_initializer %C.as.As.impl.Convert.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a89: init %i32 = splice_block %Int.as.Copy.impl.Op.call.9fd {
// CHECK:STDOUT:       %impl.elem0.88228e.1: %.737 = impl_witness_access %Copy.impl_witness.b51, element0 [concrete = %Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:       %bound_method.294: <bound method> = bound_method %.c1e, %impl.elem0.88228e.1
// CHECK:STDOUT:       %specific_fn.96a233.1: <specific function> = specific_function %impl.elem0.88228e.1, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.ea6: <bound method> = bound_method %.c1e, %specific_fn.96a233.1
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call.9fd: init %i32 = call %bound_method.ea6(%.c1e)
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
// CHECK:STDOUT:     %T.patt.loc5_20.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_20.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_29.1: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_29.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_29.1: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_29.1 [template = %x.patt.loc5_29.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete = constants.%return.param_patt.a9a]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.e1b]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_37: Core.Form = init_form %i32.loc5 [concrete = constants.%.795]
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
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
// CHECK:STDOUT:   %T.patt.loc5_20.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_20.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc5_20.1: type = symbolic_binding T, 0, template [template = %T.loc5_20.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5_20.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc5_29.2: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_29.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_29.2: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_29.2 [template = %x.patt.loc5_29.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5_20.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc6_12.2: <instruction> = convert_action %x.ref, constants.%i32, element7 [template]
// CHECK:STDOUT:   %.loc6_12.3: %i32 = splice_inst %.loc6_12.2 [template = %.loc6_12.3 (constants.%.865)]
// CHECK:STDOUT:   %.loc6_18.2: <instruction> = convert_to_category_action %.loc6_12.1, element10 [template]
// CHECK:STDOUT:   %.loc6_18.3: %i32 = splice_inst %.loc6_18.2 [template = %.loc6_18.3 (constants.%.c28)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AsI32.%T.loc5_20.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AsI32.%T.loc5_20.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_12.1: %i32 = splice_inst %.loc6_12.2 [template = %.loc6_12.3 (constants.%.865)]
// CHECK:STDOUT:     %.loc6_18.1: %i32 = splice_inst %.loc6_18.2 [template = %.loc6_18.3 (constants.%.c28)]
// CHECK:STDOUT:     return %.loc6_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsI32(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc5_20.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc5_29.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_29.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsI32(constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_20.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_29.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_29.2 => constants.%x.patt.e3b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_12.2 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_12.3 => invalid
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_18.3 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsI32(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc5_20.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_20.1 => constants.%C
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc5_29.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc5_29.2 => constants.%x.patt.953
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.cdf
// CHECK:STDOUT:   %.loc6_12.2 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_12.3 => invalid
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_18.3 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- as_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [template]
// CHECK:STDOUT:   %U.patt.cb8: %pattern_type.9a5 = symbolic_binding_pattern U, 1, template [template]
// CHECK:STDOUT:   %U.091: type = symbolic_binding U, 1, template [template]
// CHECK:STDOUT:   %AsType.type: type = fn_type @AsType [concrete]
// CHECK:STDOUT:   %AsType: %AsType.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %U.091 [template]
// CHECK:STDOUT:   %pattern_type.946: type = pattern_type %U.091 [template]
// CHECK:STDOUT:   %_.patt.972: %pattern_type.946 = value_binding_pattern _ [template]
// CHECK:STDOUT:   %.70e: %U.091 = splice_inst @AsType.%.loc6_16.3 [template]
// CHECK:STDOUT:   %.33e: %U.091 = splice_inst @AsType.%.loc6_16.5 [template]
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
// CHECK:STDOUT:   %inst.specific_inst: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.5a5: %i32 = specific_inst @AsType.%x.ref, @AsType(%i32, %i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f67: %i32 = splice_block %.5a5 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.419: type = fn_type_with_self_type %As.WithSelf.Convert.type.087, %As.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c88: %i32 = splice_block %.160 {
// CHECK:STDOUT:       %.c99: %C = specific_inst @AsType.%x.ref, @AsType(%C, %i32)
// CHECK:STDOUT:       %impl.elem0.47b: %.419 = impl_witness_access %As.impl_witness.80e, element0 [concrete = %C.as.As.impl.Convert]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.c99, %impl.elem0.47b
// CHECK:STDOUT:       %C.as.As.impl.Convert.call: init %i32 = call %bound_method(%.c99)
// CHECK:STDOUT:       %.160: %i32 = value_of_initializer %C.as.As.impl.Convert.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.7f3: %i32 = splice_block %.c88 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AsType.decl: %AsType.type = fn_decl @AsType [concrete = constants.%AsType] {
// CHECK:STDOUT:     %T.patt.loc5_21.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_21.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_30.1: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_30.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_30.1: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_30.1 [template = %x.patt.loc5_30.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %U.patt.loc5_45.1: %pattern_type.9a5 = symbolic_binding_pattern U, 1, template [template = %U.patt.loc5_45.2 (constants.%U.patt.cb8)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_23.1: type = splice_block %.loc5_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_21: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_21.2: type = symbolic_binding T, 0, template [template = %T.loc5_21.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x.param: @AsType.%T.loc5_21.1 (%T.67db0b.1) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc5_21.2 [template = %T.loc5_21.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %x: @AsType.%T.loc5_21.1 (%T.67db0b.1) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %.loc5_47.1: type = splice_block %.loc5_47.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc5_45: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc5_47.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc5_45.2: type = symbolic_binding U, 1, template [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AsType(%T.loc5_21.2: type, %U.loc5_45.2: type) {
// CHECK:STDOUT:   %T.patt.loc5_21.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_21.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc5_21.1: type = symbolic_binding T, 0, template [template = %T.loc5_21.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   %pattern_type.loc5: type = pattern_type %T.loc5_21.1 [template = %pattern_type.loc5 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc5_30.2: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_30.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc5_30.2: @AsType.%pattern_type.loc5 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_30.2 [template = %x.patt.loc5_30.2 (constants.%x.patt.260)]
// CHECK:STDOUT:   %U.patt.loc5_45.2: %pattern_type.9a5 = symbolic_binding_pattern U, 1, template [template = %U.patt.loc5_45.2 (constants.%U.patt.cb8)]
// CHECK:STDOUT:   %U.loc5_45.1: type = symbolic_binding U, 1, template [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5: <witness> = require_complete_type %T.loc5_21.1 [template = %require_complete.loc5 (constants.%require_complete.944)]
// CHECK:STDOUT:   %require_complete.loc6: <witness> = require_complete_type %U.loc5_45.1 [template = %require_complete.loc6 (constants.%require_complete.441)]
// CHECK:STDOUT:   %pattern_type.loc6: type = pattern_type %U.loc5_45.1 [template = %pattern_type.loc6 (constants.%pattern_type.946)]
// CHECK:STDOUT:   %_.patt.loc6_8.2: @AsType.%pattern_type.loc6 (%pattern_type.946) = value_binding_pattern _ [template = %_.patt.loc6_8.2 (constants.%_.patt.972)]
// CHECK:STDOUT:   %.loc6_16.3: <instruction> = convert_action %x.ref, %U.loc5_45.1, element7 [template]
// CHECK:STDOUT:   %.loc6_16.4: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.3 [template = %.loc6_16.4 (constants.%.70e)]
// CHECK:STDOUT:   %.loc6_16.5: <instruction> = convert_to_category_action %.loc6_16.1, element1 [template]
// CHECK:STDOUT:   %.loc6_16.6: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.5 [template = %.loc6_16.6 (constants.%.33e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AsType.%T.loc5_21.1 (%T.67db0b.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AsType.%T.loc5_21.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %U.ref.loc6_19: type = name_ref U, %U.loc5_45.2 [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:     %.loc6_16.1: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.3 [template = %.loc6_16.4 (constants.%.70e)]
// CHECK:STDOUT:     %.loc6_16.2: @AsType.%U.loc5_45.1 (%U.091) = splice_inst %.loc6_16.5 [template = %.loc6_16.6 (constants.%.33e)]
// CHECK:STDOUT:     %U.ref.loc6_10: type = name_ref U, %U.loc5_45.2 [template = %U.loc5_45.1 (constants.%U.091)]
// CHECK:STDOUT:     %_: @AsType.%U.loc5_45.1 (%U.091) = wrapper_binding _, %.loc6_16.2
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %_.patt.loc6_8.1: @AsType.%pattern_type.loc6 (%pattern_type.946) = value_binding_pattern _ [template = %_.patt.loc6_8.2 (constants.%_.patt.972)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsType(constants.%T.67db0b.1, constants.%U.091) {
// CHECK:STDOUT:   %T.patt.loc5_21.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_21.1 => constants.%T.67db0b.1
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc5_30.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc5_30.2 => constants.%x.patt.260
// CHECK:STDOUT:   %U.patt.loc5_45.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc5_45.1 => constants.%U.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsType(constants.%i32, constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_21.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_21.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_30.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_30.2 => constants.%x.patt.e3b
// CHECK:STDOUT:   %U.patt.loc5_45.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc5_45.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %_.patt.loc6_8.2 => constants.%_.patt.bed
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_16.4 => invalid
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_16.6 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AsType(constants.%C, constants.%i32) {
// CHECK:STDOUT:   %T.patt.loc5_21.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc5_21.1 => constants.%C
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.98b
// CHECK:STDOUT:   %x.param_patt.loc5_30.2 => constants.%x.param_patt.0f9
// CHECK:STDOUT:   %x.patt.loc5_30.2 => constants.%x.patt.953
// CHECK:STDOUT:   %U.patt.loc5_45.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc5_45.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5 => constants.%complete_type.cdf
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.6b6
// CHECK:STDOUT:   %_.patt.loc6_8.2 => constants.%_.patt.bed
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_16.4 => invalid
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_16.6 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

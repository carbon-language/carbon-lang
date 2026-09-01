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
fn F(template T: Core.Destroy) {
  var unused w: T = 0;
}
//@dump-sem-ir-end

fn G() {
  F(i32);
}

// --- initialize_primitive.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn F[template T: Core.Destroy](x: T) {
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
fn F(template T: Core.Destroy) {
  var unused w: T = (Make(), Make(), Make());
}
//@dump-sem-ir-end

class Y {}

impl (X, X, X) as Core.ImplicitAs(Y) {
  // CHECK:STDERR: initialize_from_tuple.carbon:[[@LINE+4]]:14: warning: binding `self` unused [UnusedBinding]
  // CHECK:STDERR:   fn Convert(self: (X, X, X)) -> Y { return {}; }
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn Convert(self: (X, X, X)) -> Y { return {}; }
}

fn G() {
  F((X, X, X));

  F(array(X, 3));

  F(Y);
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
// CHECK:STDOUT:   %.loc6_16.2 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_16.3 => invalid
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
// CHECK:STDOUT:   %.loc6_16.2 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc6_16.3 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- initialize_dependent_target_from_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: %Destroy.type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [template]
// CHECK:STDOUT:   %require_complete.14c: <witness> = require_complete_type %T.as_type [template]
// CHECK:STDOUT:   %pattern_type.f4e485.2: type = pattern_type %T.as_type [template]
// CHECK:STDOUT:   %w.patt.4ab: %pattern_type.f4e485.2 = ref_binding_pattern w [template]
// CHECK:STDOUT:   %w.var_patt.97a: %pattern_type.f4e485.2 = var_pattern %w.patt.4ab [template]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.f89: %T.as_type = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc6_3.11 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc6_3.11 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc11_8.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6c4ec3.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.6de: %Destroy.type = facet_value %i32, (%custom_witness.6c4ec3.2) [concrete]
// CHECK:STDOUT:   %w.patt.859: %pattern_type.6b6 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.82c: %pattern_type.6b6 = var_pattern %w.patt.859 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.170: type = fn_type @Destroy.WithSelf.Op.1, @Destroy.WithSelf(%Destroy.facet.6de) [concrete]
// CHECK:STDOUT:   %.d91: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.170, %Destroy.facet.6de [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.eea: <bound method> = splice_block %bound_method.90f {
// CHECK:STDOUT:       %.2f9: ref %i32 = specific_inst @F.%w.var, @F(%Destroy.facet.6de)
// CHECK:STDOUT:       %impl.elem0.5ef: %.d91 = impl_witness_access %custom_witness.6c4ec3.2, element0 [concrete = %Destroy.WithSelf.Op.403171.2]
// CHECK:STDOUT:       %bound_method.90f: <bound method> = bound_method %.2f9, %impl.elem0.5ef
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %.eea(%.2f9)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1aa = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: %Destroy.type) {
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_16.1: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc6_17.2: type = facet_access_type %T.loc5_16.1 [template = %T.as_type.loc6_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc6_17.2 [template = %require_complete (constants.%require_complete.14c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc6_17.2 [template = %pattern_type (constants.%pattern_type.f4e485.2)]
// CHECK:STDOUT:   %w.patt.loc6_15.2: @F.%pattern_type (%pattern_type.f4e485.2) = ref_binding_pattern w [template = %w.patt.loc6_15.2 (constants.%w.patt.4ab)]
// CHECK:STDOUT:   %w.var_patt.loc6_3.2: @F.%pattern_type (%pattern_type.f4e485.2) = var_pattern %w.patt.loc6_15.2 [template = %w.var_patt.loc6_3.2 (constants.%w.var_patt.97a)]
// CHECK:STDOUT:   %.loc6_3.6: %tuple.type = initialize_action %int_0, %T.as_type.loc6_17.2, %w.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc6_3.6, element0 [template]
// CHECK:STDOUT:   %.loc6_3.7: @F.%T.as_type.loc6_17.2 (%T.as_type) = splice_inst %tuple.elem0 [template = %.loc6_3.7 (constants.%.f89)]
// CHECK:STDOUT:   %.loc6_3.8: <instruction> = compound_member_access_action %w.var, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc6_3.9: type = type_of_inst %.loc6_3.8 [template = %.loc6_3.9 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc6_3.10: @F.%.loc6_3.9 (%.8cb) = splice_inst %.loc6_3.8 [template = %.loc6_3.10 (constants.%.c23)]
// CHECK:STDOUT:   %.loc6_3.11: <instruction> = call_action (%.loc6_3.3), true [template]
// CHECK:STDOUT:   %.loc6_3.12: type = type_of_inst %.loc6_3.11 [template = %.loc6_3.12 (constants.%.57c)]
// CHECK:STDOUT:   %.loc6_3.13: @F.%.loc6_3.12 (%.57c) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref @F.%T.as_type.loc6_17.2 (%T.as_type) = var_storage %w.var_patt.loc6_3.1
// CHECK:STDOUT:     %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:     %.loc6_3.1: @F.%T.as_type.loc6_17.2 (%T.as_type) = splice_inst %tuple.elem0 [template = %.loc6_3.7 (constants.%.f89)]
// CHECK:STDOUT:     assign %w.var, %.loc6_3.1
// CHECK:STDOUT:     %.loc6_17.1: type = splice_block %.loc6_17.2 [template = %T.as_type.loc6_17.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Destroy.type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc6_17.1: type = facet_access_type %T.ref [template = %T.as_type.loc6_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc6_17.2: type = converted %T.ref, %T.as_type.loc6_17.1 [template = %T.as_type.loc6_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%T.as_type.loc6_17.2 (%T.as_type) = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt.loc6_15.1: @F.%pattern_type (%pattern_type.f4e485.2) = ref_binding_pattern w [template = %w.patt.loc6_15.2 (constants.%w.patt.4ab)]
// CHECK:STDOUT:       %w.var_patt.loc6_3.1: @F.%pattern_type (%pattern_type.f4e485.2) = var_pattern %w.patt.loc6_15.1 [template = %w.var_patt.loc6_3.2 (constants.%w.var_patt.97a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6_3.2: type = type_of_inst %.loc6_3.8 [template = %.loc6_3.9 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc6_3.3: @F.%.loc6_3.9 (%.8cb) = splice_inst %.loc6_3.8 [template = %.loc6_3.10 (constants.%.c23)]
// CHECK:STDOUT:     %.loc6_3.4: type = type_of_inst %.loc6_3.11 [template = %.loc6_3.12 (constants.%.57c)]
// CHECK:STDOUT:     %.loc6_3.5: @F.%.loc6_3.12 (%.57c) = splice_inst %.loc6_3.11 [template = %.loc6_3.13 (constants.%.e63)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Destroy.facet.6de) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%Destroy.facet.6de
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc6_17.2 => constants.%i32
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: %Destroy.type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [template]
// CHECK:STDOUT:   %pattern_type.f4e485.2: type = pattern_type %T.as_type [template]
// CHECK:STDOUT:   %x.param_patt.dd4: %pattern_type.f4e485.2 = value_param_pattern [template]
// CHECK:STDOUT:   %x.patt.845: %pattern_type.f4e485.2 = wrapper_binding_pattern x, %x.param_patt.dd4 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.14c: <witness> = require_complete_type %T.as_type [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %w.patt: %pattern_type.6b6 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt: %pattern_type.6b6 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.a11: %i32 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc6_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.6c4ec3.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.6de: %Destroy.type = facet_value %i32, (%custom_witness.6c4ec3.2) [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %x.param_patt.cee: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.e3b: %pattern_type.6b6 = wrapper_binding_pattern x, %x.param_patt.cee [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.8cb5e6.1: type = type_of_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template]
// CHECK:STDOUT:   %.c234ea.1: %.8cb5e6.1 = splice_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template]
// CHECK:STDOUT:   %.57c638.1: type = type_of_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template]
// CHECK:STDOUT:   %.e63d45.1: %.57c638.1 = splice_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template]
// CHECK:STDOUT:   %.254040.1: %.57c638.1 = splice_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.53f: %.57c638.1 = splice_block %.254040.2 [template = %.254040.1] {
// CHECK:STDOUT:       %.8cb5e6.2: type = type_of_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template = %.8cb5e6.1]
// CHECK:STDOUT:       %.c234ea.2: %.8cb5e6.1 = splice_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template = %.c234ea.1]
// CHECK:STDOUT:       %.57c638.2: type = type_of_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template = %.57c638.1]
// CHECK:STDOUT:       %.e63d45.2: %.57c638.1 = splice_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template = %.e63d45.1]
// CHECK:STDOUT:       %.254040.2: %.57c638.1 = splice_inst <unexpected>.inst{{[0-9A-F]+}}.loc6_3 [template = %.254040.1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//prelude/parts/as, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.param_patt.loc5_33.1: @F.%pattern_type (%pattern_type.f4e485.2) = value_param_pattern [template = %x.param_patt.loc5_33.2 (constants.%x.param_patt.dd4)]
// CHECK:STDOUT:     %x.patt.loc5_33.1: @F.%pattern_type (%pattern_type.f4e485.2) = wrapper_binding_pattern x, %x.param_patt.loc5_33.1 [template = %x.patt.loc5_33.2 (constants.%x.patt.845)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_22: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_16.2: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.as_type.loc5_35.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_35.1: type = splice_block %.loc5_35.2 [template = %T.as_type.loc5_35.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Destroy.type = name_ref T, %T.loc5_16.2 [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc5_35.2: type = facet_access_type %T.ref [template = %T.as_type.loc5_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_35.2: type = converted %T.ref, %T.as_type.loc5_35.2 [template = %T.as_type.loc5_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%T.as_type.loc5_35.1 (%T.as_type) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc5_16.2: %Destroy.type) {
// CHECK:STDOUT:   %T.patt.loc5_16.2: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc5_16.1: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc5_16.1 (constants.%T)]
// CHECK:STDOUT:   %T.as_type.loc5_35.1: type = facet_access_type %T.loc5_16.1 [template = %T.as_type.loc5_35.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc5_35.1 [template = %pattern_type (constants.%pattern_type.f4e485.2)]
// CHECK:STDOUT:   %x.param_patt.loc5_33.2: @F.%pattern_type (%pattern_type.f4e485.2) = value_param_pattern [template = %x.param_patt.loc5_33.2 (constants.%x.param_patt.dd4)]
// CHECK:STDOUT:   %x.patt.loc5_33.2: @F.%pattern_type (%pattern_type.f4e485.2) = wrapper_binding_pattern x, %x.param_patt.loc5_33.2 [template = %x.patt.loc5_33.2 (constants.%x.patt.845)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc5_35.1 [template = %require_complete (constants.%require_complete.14c)]
// CHECK:STDOUT:   %.loc6_3.2: %tuple.type = initialize_action %x.ref, constants.%i32, %w.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc6_3.2, element0 [template]
// CHECK:STDOUT:   %.loc6_3.3: %i32 = splice_inst %tuple.elem0 [template = %.loc6_3.3 (constants.%.a11)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.as_type.loc5_35.1 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref %i32 = var_storage %w.var_patt
// CHECK:STDOUT:     %x.ref: @F.%T.as_type.loc5_35.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_3.1: %i32 = splice_inst %tuple.elem0 [template = %.loc6_3.3 (constants.%.a11)]
// CHECK:STDOUT:     assign %w.var, %.loc6_3.1
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %w: ref %i32 = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %pattern_type.6b6 = ref_binding_pattern w [concrete = constants.%w.patt]
// CHECK:STDOUT:       %w.var_patt: %pattern_type.6b6 = var_pattern %w.patt [concrete = constants.%w.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.WithSelf.Op.bound: <bound method> = bound_method %w.var, constants.%Destroy.WithSelf.Op.403171.2
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %Destroy.WithSelf.Op.bound(%w.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6_3.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%T
// CHECK:STDOUT:   %T.as_type.loc5_35.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f4e485.2
// CHECK:STDOUT:   %x.param_patt.loc5_33.2 => constants.%x.param_patt.dd4
// CHECK:STDOUT:   %x.patt.loc5_33.2 => constants.%x.patt.845
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Destroy.facet.6de) {
// CHECK:STDOUT:   %T.patt.loc5_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc5_16.1 => constants.%Destroy.facet.6de
// CHECK:STDOUT:   %T.as_type.loc5_35.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:   %x.param_patt.loc5_33.2 => constants.%x.param_patt.cee
// CHECK:STDOUT:   %x.patt.loc5_33.2 => constants.%x.patt.e3b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_3.2 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc6_3.3 => constants.%.254040.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- initialize_from_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Make.type: type = fn_type @Make [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Make: %Make.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.429: type = pattern_type %Destroy.type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: %Destroy.type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T [template]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type [template]
// CHECK:STDOUT:   %pattern_type.f4e485.2: type = pattern_type %T.as_type [template]
// CHECK:STDOUT:   %w.patt.4ab: %pattern_type.f4e485.2 = ref_binding_pattern w [template]
// CHECK:STDOUT:   %w.var_patt.97a: %pattern_type.f4e485.2 = var_pattern %w.patt.4ab [template]
// CHECK:STDOUT:   %tuple.type.92e: type = tuple_type (%X, %X, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.f39: type = tuple_type (<instruction>, <instruction>, <instruction>, <instruction>) [concrete]
// CHECK:STDOUT:   %.345: %X = splice_inst @F.%tuple.elem1 [template]
// CHECK:STDOUT:   %.595: %X = splice_inst @F.%tuple.elem2 [template]
// CHECK:STDOUT:   %.f5b: %X = splice_inst @F.%tuple.elem3 [template]
// CHECK:STDOUT:   %.149: %T.as_type = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.836: type = facet_type <@ImplicitAs, @ImplicitAs(%Y)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @tuple.type.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.eca: type = pattern_type %tuple.type.92e [concrete]
// CHECK:STDOUT:   %pattern_type.7ab: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %tuple.type.as.ImplicitAs.impl.Convert: %tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.836 = facet_value %tuple.type.92e, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.354: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Y, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc24_14.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.3: <witness> = custom_witness (%Destroy.Op.1a2547.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.6fd: %Destroy.type = facet_value %tuple.type.92e, (%custom_witness.df9cc1.3) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3, %X [concrete]
// CHECK:STDOUT:   %pattern_type.92a: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc26 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.4: <witness> = custom_witness (%Destroy.Op.1a2547.4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.f6d: %Destroy.type = facet_value %array_type, (%custom_witness.df9cc1.4) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc28 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.5: <witness> = custom_witness (%Destroy.Op.1a2547.5), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.858: %Destroy.type = facet_value %Y, (%custom_witness.df9cc1.5) [concrete]
// CHECK:STDOUT:   %complete_type.8ad: <witness> = complete_type_witness %tuple.type.92e [concrete]
// CHECK:STDOUT:   %w.patt.689: %pattern_type.eca = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.e50: %pattern_type.eca = var_pattern %w.patt.689 [concrete]
// CHECK:STDOUT:   %inst.tuple_init: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.262: init %tuple.type.92e to @F.%w.var = tuple_init (@F.%Make.call.loc9_27, @F.%Make.call.loc9_35, @F.%Make.call.loc9_43)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem0: ref %X = tuple_access @F.%w.var, element0
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem1: ref %X = tuple_access @F.%w.var, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem2: ref %X = tuple_access @F.%w.var, element2
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.1: %tuple.type.f39 = tuple_value (%inst.tuple_init, %inst.tuple_access.426619.1, %inst.tuple_access.426619.2, %inst.tuple_access.426619.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ae9: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.6fd) [concrete]
// CHECK:STDOUT:   %.2a3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ae9, %Destroy.facet.6fd [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c33: <bound method> = splice_block %bound_method.6a0 {
// CHECK:STDOUT:       %.45e: ref %tuple.type.92e = specific_inst @F.%w.var, @F(%Destroy.facet.6fd)
// CHECK:STDOUT:       %impl.elem0.2e2: %.2a3 = impl_witness_access %custom_witness.df9cc1.3, element0 [concrete = %Destroy.Op.1a2547.3]
// CHECK:STDOUT:       %bound_method.6a0: <bound method> = bound_method %.45e, %impl.elem0.2e2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.Op.call.7f1: init %empty_tuple.type = call %.c33(%.45e)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.3e8: <witness> = complete_type_witness %array_type [concrete]
// CHECK:STDOUT:   %w.patt.f13: %pattern_type.92a = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.c83: %pattern_type.92a = var_pattern %w.patt.f13 [concrete]
// CHECK:STDOUT:   %int_0.5c61a1.1: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %int_1.5b8115.1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc8af.1: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %inst.array_init: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.27e: init %array_type to @F.%w.var = array_init (@F.%Make.call.loc9_27, @F.%Make.call.loc9_35, @F.%Make.call.loc9_43)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.16c: ref %X = splice_block %.25f {
// CHECK:STDOUT:       %int_0.5c61a1.2: Core.IntLiteral = int_value 0 [concrete = %int_0.5c61a1.1]
// CHECK:STDOUT:       %.25f: ref %X = array_index @F.%w.var, %int_0.5c61a1.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.000: ref %X = splice_block %.532 {
// CHECK:STDOUT:       %int_1.5b8115.2: Core.IntLiteral = int_value 1 [concrete = %int_1.5b8115.1]
// CHECK:STDOUT:       %.532: ref %X = array_index @F.%w.var, %int_1.5b8115.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.def: ref %X = splice_block %.08a {
// CHECK:STDOUT:       %int_2.ecc8af.2: Core.IntLiteral = int_value 2 [concrete = %int_2.ecc8af.1]
// CHECK:STDOUT:       %.08a: ref %X = array_index @F.%w.var, %int_2.ecc8af.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.2: %tuple.type.f39 = tuple_value (%inst.array_init, %inst.splice_block.426619.2, %inst.splice_block.426619.3, %inst.splice_block.426619.4) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ca5: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.f6d) [concrete]
// CHECK:STDOUT:   %.2aa: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.ca5, %Destroy.facet.f6d [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.af9: <bound method> = splice_block %bound_method.cc6 {
// CHECK:STDOUT:       %.ae0: ref %array_type = specific_inst @F.%w.var, @F(%Destroy.facet.f6d)
// CHECK:STDOUT:       %impl.elem0.08d: %.2aa = impl_witness_access %custom_witness.df9cc1.4, element0 [concrete = %Destroy.Op.1a2547.4]
// CHECK:STDOUT:       %bound_method.cc6: <bound method> = bound_method %.ae0, %impl.elem0.08d
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.Op.call.186: init %empty_tuple.type = call %.af9(%.ae0)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.patt.22a: %pattern_type.7ab = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.565: %pattern_type.7ab = var_pattern %w.patt.22a [concrete]
// CHECK:STDOUT:   %.e93: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.354, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.97c: init %Y = splice_block %tuple.type.as.ImplicitAs.impl.Convert.call {
// CHECK:STDOUT:       %impl.elem0.bd3: %.e93 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.28b: <bound method> = bound_method @F.%.loc9_44, %impl.elem0.bd3
// CHECK:STDOUT:       %.ba4: ref %T.as_type = splice_block @F.%w.var {}
// CHECK:STDOUT:       %.f07: ref %X = temporary @F.%.loc9_3.1, @F.%Make.call.loc9_27
// CHECK:STDOUT:       %.60d: %X = acquire_value %.f07
// CHECK:STDOUT:       %.127: ref %X = temporary @F.%.loc9_3.2, @F.%Make.call.loc9_35
// CHECK:STDOUT:       %.7c6: %X = acquire_value %.127
// CHECK:STDOUT:       %.c35: ref %X = temporary @F.%.loc9_3.3, @F.%Make.call.loc9_43
// CHECK:STDOUT:       %.d44: %X = acquire_value %.c35
// CHECK:STDOUT:       %tuple.675: %tuple.type.92e = tuple_value (%.60d, %.7c6, %.d44)
// CHECK:STDOUT:       %.294: %tuple.type.92e = converted @F.%.loc9_44, %tuple.675
// CHECK:STDOUT:       %tuple.type.as.ImplicitAs.impl.Convert.call: init %Y to %.ba4 = call %bound_method.28b(%.294)
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
// CHECK:STDOUT:   %tuple.1ed7a0.3: %tuple.type.f39 = tuple_value (%inst.splice_block.426619.6, %inst.temporary_storage.426619.1, %inst.temporary_storage.426619.2, %inst.temporary_storage.426619.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.b06: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.858) [concrete]
// CHECK:STDOUT:   %.24d: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.b06, %Destroy.facet.858 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.63e: <bound method> = splice_block %bound_method.e05 {
// CHECK:STDOUT:       %.5be: ref %Y = specific_inst @F.%w.var, @F(%Destroy.facet.858)
// CHECK:STDOUT:       %impl.elem0.01e: %.24d = impl_witness_access %custom_witness.df9cc1.5, element0 [concrete = %Destroy.Op.1a2547.5]
// CHECK:STDOUT:       %bound_method.e05: <bound method> = bound_method %.5be, %impl.elem0.01e
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.Op.call.160: init %empty_tuple.type = call %.63e(%.5be)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc8_16.1: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8: type = splice_block %Destroy.ref [concrete = constants.%Destroy.type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_16.2: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_16.2: %Destroy.type) {
// CHECK:STDOUT:   %T.patt.loc8_16.2: %pattern_type.429 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_16.1: %Destroy.type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_17.2: type = facet_access_type %T.loc8_16.1 [template = %T.as_type.loc9_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc9_17.2 [template = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_17.2 [template = %pattern_type (constants.%pattern_type.f4e485.2)]
// CHECK:STDOUT:   %w.patt.loc9_15.2: @F.%pattern_type (%pattern_type.f4e485.2) = ref_binding_pattern w [template = %w.patt.loc9_15.2 (constants.%w.patt.4ab)]
// CHECK:STDOUT:   %w.var_patt.loc9_3.2: @F.%pattern_type (%pattern_type.f4e485.2) = var_pattern %w.patt.loc9_15.2 [template = %w.var_patt.loc9_3.2 (constants.%w.var_patt.97a)]
// CHECK:STDOUT:   %.loc9_3.9: %tuple.type.f39 = initialize_action %.loc9_44, %T.as_type.loc9_17.2, %w.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc9_3.9, element0 [template]
// CHECK:STDOUT:   %tuple.elem1: <instruction> = tuple_access %.loc9_3.9, element1 [template]
// CHECK:STDOUT:   %.loc9_3.10: %X = splice_inst %tuple.elem1 [template = %.loc9_3.10 (constants.%.345)]
// CHECK:STDOUT:   %tuple.elem2: <instruction> = tuple_access %.loc9_3.9, element2 [template]
// CHECK:STDOUT:   %.loc9_3.11: %X = splice_inst %tuple.elem2 [template = %.loc9_3.11 (constants.%.595)]
// CHECK:STDOUT:   %tuple.elem3: <instruction> = tuple_access %.loc9_3.9, element3 [template]
// CHECK:STDOUT:   %.loc9_3.12: %X = splice_inst %tuple.elem3 [template = %.loc9_3.12 (constants.%.f5b)]
// CHECK:STDOUT:   %.loc9_3.13: @F.%T.as_type.loc9_17.2 (%T.as_type) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.149)]
// CHECK:STDOUT:   %.loc9_3.14: <instruction> = compound_member_access_action %w.var, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc9_3.15: type = type_of_inst %.loc9_3.14 [template = %.loc9_3.15 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc9_3.16: @F.%.loc9_3.15 (%.8cb) = splice_inst %.loc9_3.14 [template = %.loc9_3.16 (constants.%.c23)]
// CHECK:STDOUT:   %.loc9_3.17: <instruction> = call_action (%.loc9_3.6), true [template]
// CHECK:STDOUT:   %.loc9_3.18: type = type_of_inst %.loc9_3.17 [template = %.loc9_3.18 (constants.%.57c)]
// CHECK:STDOUT:   %.loc9_3.19: @F.%.loc9_3.18 (%.57c) = splice_inst %.loc9_3.17 [template = %.loc9_3.19 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref @F.%T.as_type.loc9_17.2 (%T.as_type) = var_storage %w.var_patt.loc9_3.1
// CHECK:STDOUT:     %Make.ref.loc9_22: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.1: %X = splice_inst %tuple.elem1 [template = %.loc9_3.10 (constants.%.345)]
// CHECK:STDOUT:     %Make.call.loc9_27: init %X to %.loc9_3.1 = call %Make.ref.loc9_22()
// CHECK:STDOUT:     %Make.ref.loc9_30: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.2: %X = splice_inst %tuple.elem2 [template = %.loc9_3.11 (constants.%.595)]
// CHECK:STDOUT:     %Make.call.loc9_35: init %X to %.loc9_3.2 = call %Make.ref.loc9_30()
// CHECK:STDOUT:     %Make.ref.loc9_38: %Make.type = name_ref Make, file.%Make.decl [concrete = constants.%Make]
// CHECK:STDOUT:     %.loc9_3.3: %X = splice_inst %tuple.elem3 [template = %.loc9_3.12 (constants.%.f5b)]
// CHECK:STDOUT:     %Make.call.loc9_43: init %X to %.loc9_3.3 = call %Make.ref.loc9_38()
// CHECK:STDOUT:     %.loc9_44: %tuple.type.92e = tuple_literal (%Make.call.loc9_27, %Make.call.loc9_35, %Make.call.loc9_43)
// CHECK:STDOUT:     %.loc9_3.4: @F.%T.as_type.loc9_17.2 (%T.as_type) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.149)]
// CHECK:STDOUT:     assign %w.var, %.loc9_3.4
// CHECK:STDOUT:     %.loc9_17.1: type = splice_block %.loc9_17.2 [template = %T.as_type.loc9_17.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Destroy.type = name_ref T, %T.loc8_16.2 [template = %T.loc8_16.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type.loc9_17.1: type = facet_access_type %T.ref [template = %T.as_type.loc9_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_17.2: type = converted %T.ref, %T.as_type.loc9_17.1 [template = %T.as_type.loc9_17.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %w: ref @F.%T.as_type.loc9_17.2 (%T.as_type) = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt.loc9_15.1: @F.%pattern_type (%pattern_type.f4e485.2) = ref_binding_pattern w [template = %w.patt.loc9_15.2 (constants.%w.patt.4ab)]
// CHECK:STDOUT:       %w.var_patt.loc9_3.1: @F.%pattern_type (%pattern_type.f4e485.2) = var_pattern %w.patt.loc9_15.1 [template = %w.var_patt.loc9_3.2 (constants.%w.var_patt.97a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_3.5: type = type_of_inst %.loc9_3.14 [template = %.loc9_3.15 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc9_3.6: @F.%.loc9_3.15 (%.8cb) = splice_inst %.loc9_3.14 [template = %.loc9_3.16 (constants.%.c23)]
// CHECK:STDOUT:     %.loc9_3.7: type = type_of_inst %.loc9_3.17 [template = %.loc9_3.18 (constants.%.57c)]
// CHECK:STDOUT:     %.loc9_3.8: @F.%.loc9_3.18 (%.57c) = splice_inst %.loc9_3.17 [template = %.loc9_3.19 (constants.%.e63)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Destroy.facet.6fd) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%Destroy.facet.6fd
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_17.2 => constants.%tuple.type.92e
// CHECK:STDOUT:   %require_complete => constants.%complete_type.8ad
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.eca
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.689
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.e50
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.1
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.tuple_init
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.tuple_access.426619.1
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.tuple_access.426619.2
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.tuple_access.426619.3
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => invalid
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.1
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Destroy.facet.f6d) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%Destroy.facet.f6d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_17.2 => constants.%array_type
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3e8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.92a
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.f13
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.c83
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.2
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.array_init
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.splice_block.426619.4
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
// CHECK:STDOUT: specific @F(constants.%Destroy.facet.858) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%Destroy.facet.858
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc9_17.2 => constants.%Y
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ab
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.22a
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.565
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.3
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.temporary_storage.426619.1
// CHECK:STDOUT:   %.loc9_3.10 => invalid
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.temporary_storage.426619.2
// CHECK:STDOUT:   %.loc9_3.11 => invalid
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.temporary_storage.426619.3
// CHECK:STDOUT:   %.loc9_3.12 => invalid
// CHECK:STDOUT:   %.loc9_3.13 => invalid
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.7
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
// CHECK:STDOUT:   %.loc6_12.2 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_12.3 => invalid
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.426619.1
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
// CHECK:STDOUT:   %.loc6_12.2 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_12.3 => invalid
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.426619.3
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
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_16.4 => invalid
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_16.6 => invalid
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
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_16.4 => invalid
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_16.6 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:

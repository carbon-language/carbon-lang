// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
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

// --- splice_inst_storage_arg.carbon

library "[[@TEST_NAME]]";

class B {
  fn MakeB() -> B {
    return {};
  }
}

class A(template T: type) {
  //@dump-sem-ir-begin
  fn MakeA() -> A(T) {
    return {.field = T.MakeB()};
  }
  //@dump-sem-ir-end

  var field: T;
}

fn F() {
  //@dump-sem-ir-begin
  A(B).MakeA();
  //@dump-sem-ir-end
}

// --- initialize_from_template_method_call.carbon

library "[[@TEST_NAME]]";

class B {
  fn MakeB() -> B {
    return {};
  }
}

//@dump-sem-ir-begin
class A {
  var field: B;
}

fn MakeA(template T: type) -> A {
   return {
     .field = T.MakeB(),
   };
}

fn Run() {
  MakeA(B);
}
//@dump-sem-ir-end

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
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %n.patt.b7d: %pattern_type.9d9 = value_binding_pattern n [concrete]
// CHECK:STDOUT:   %.ceb: %i32 = splice_inst @F.%.loc6_16.2 [template]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %x.param_patt.3bc: %pattern_type.9d9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0ff: %pattern_type.9d9 = wrapper_binding_pattern x, %x.param_patt.3bc [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ff9: <witness> = impl_witness @C.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert.type: type = fn_type @C.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %C.as.ImplicitAs.impl.Convert: %C.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value %C, (%ImplicitAs.impl_witness.ff9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.95e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.26a: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %inst.specific_inst: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.25f: %i32 = specific_inst @F.%x.ref, @F(%i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.f6c: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.95e, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c39: %i32 = splice_block %.bbe {
// CHECK:STDOUT:       %.340: %C = specific_inst @F.%x.ref, @F(%C)
// CHECK:STDOUT:       %impl.elem0.597: %.f6c = impl_witness_access %ImplicitAs.impl_witness.ff9, element0 [concrete = %C.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.340, %impl.elem0.597
// CHECK:STDOUT:       %C.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method(%.340)
// CHECK:STDOUT:       %.bbe: %i32 = value_of_initializer %C.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc5_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_25.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_25.1: @F.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_25.1 [template = %x.patt.loc5_25.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_33: Core.Form = init_form %i32.loc5 [concrete = constants.%.55c]
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
// CHECK:STDOUT:   %.loc6_16.3: %i32 = splice_inst %.loc6_16.2 [template = %.loc6_16.3 (constants.%.ceb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_16.1: %i32 = splice_inst %.loc6_16.2 [template = %.loc6_16.3 (constants.%.ceb)]
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %n: %i32 = wrapper_binding n, %.loc6_16.1
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %n.patt: %pattern_type.9d9 = value_binding_pattern n [concrete = constants.%n.patt.b7d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %n.ref: %i32 = name_ref n, %n
// CHECK:STDOUT:     %impl.elem0: %.481 = impl_witness_access constants.%Copy.impl_witness.0ff, element0 [concrete = constants.%Int.as.Copy.impl.Op.617]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.3bc
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.0ff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_16.2 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_16.3 => <not constant>
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.26a
// CHECK:STDOUT:   %.loc6_16.2 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc6_16.3 => <not constant>
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.869: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.c97: %Destroy.WithSelf.SelfDestruct.type.869 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.df0 [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc6_3.8 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc6_3.11 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc6_3.11 [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %w.patt.753: %pattern_type.9d9 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.1d3: %pattern_type.9d9 = var_pattern %w.patt.753 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.09b: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.c85: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec2: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dd1, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f16 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.09b = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec2) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e04: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.f0f: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.e04, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.614: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.2b2: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.943: init %i32 = splice_block %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = %int_0.2b2] {
// CHECK:STDOUT:       %impl.elem0.1cb: %.f0f = impl_witness_access %ImplicitAs.impl_witness.ec2, element0 [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.ba1]
// CHECK:STDOUT:       %bound_method.711: <bound method> = bound_method %int_0.5c6, %impl.elem0.1cb [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.1cb, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete = %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:       %bound_method.bf5: <bound method> = bound_method %int_0.5c6, %specific_fn [concrete = %bound_method.614]
// CHECK:STDOUT:       %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.bf5(%int_0.5c6) [concrete = %int_0.2b2]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.1) [concrete]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f: <witness> = custom_witness (%Destroy.WithSelf.Op.403, %Destroy.WithSelf.SubobjectDestroy.d01, %Destroy.WithSelf.SelfDestruct.db3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2b3: %Destroy.type = facet_value %i32, (%custom_witness.f8f) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.357: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.2b3) [concrete]
// CHECK:STDOUT:   %.00f: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.357, %Destroy.facet.2b3 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.cda: <bound method> = splice_block %bound_method.8eb {
// CHECK:STDOUT:       %.431: ref %i32 = specific_inst @F.%w.var, @F(%i32)
// CHECK:STDOUT:       %impl.elem2: %.00f = impl_witness_access %custom_witness.f8f, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3]
// CHECK:STDOUT:       %bound_method.8eb: <bound method> = bound_method %.431, %impl.elem2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %.cda(%.431)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.df0: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.869) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.c97)]
// CHECK:STDOUT:   %Core.import_ref.09d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.cb9) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.c85)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dd1 = impl_witness_table (%Core.import_ref.09d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc6 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
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
// CHECK:STDOUT:   %.loc6_3.6: %tuple.type = initialize_action %int_0, %T.loc5_16.1, %w.var, none, false [template]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT:   %w.patt.loc6_15.2 => constants.%w.patt.753
// CHECK:STDOUT:   %w.var_patt.loc6_3.2 => constants.%w.var_patt.1d3
// CHECK:STDOUT:   %.loc6_3.6 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_3.7 => constants.%int_0.2b2
// CHECK:STDOUT:   %.loc6_3.8 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_3.9 => <bound method>
// CHECK:STDOUT:   %.loc6_3.10 => <not constant>
// CHECK:STDOUT:   %.loc6_3.11 => constants.%inst.call
// CHECK:STDOUT:   %.loc6_3.12 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc6_3.13 => <not constant>
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
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %w.patt: %pattern_type.9d9 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt: %pattern_type.9d9 = var_pattern %w.patt [concrete]
// CHECK:STDOUT:   %tuple.type.f7a: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.721: %i32 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %self.param_patt.6c0: %pattern_type.9d9 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.1ea: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt.6c0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfc: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01: %Destroy.WithSelf.SubobjectDestroy.type.dfc = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef0: type = fn_type @Destroy.WithSelf.Op.loc6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403: %Destroy.WithSelf.Op.type.ef0 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbc: type = fn_type @Destroy.WithSelf.SelfDestruct.loc6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3: %Destroy.WithSelf.SelfDestruct.type.fbc = struct_value () [concrete]
// CHECK:STDOUT:   %x.param_patt.3bc: %pattern_type.9d9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0ff: %pattern_type.9d9 = wrapper_binding_pattern x, %x.param_patt.3bc [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %inst.splice_block: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d58: init %i32 = splice_block %Int.as.Copy.impl.Op.call {
// CHECK:STDOUT:       %.25f: %i32 = specific_inst @F.%x.ref, @F(%i32)
// CHECK:STDOUT:       %impl.elem0.b1a: %.481 = impl_witness_access %Copy.impl_witness.0ff, element0 [concrete = %Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:       %bound_method.7b6: <bound method> = bound_method %.25f, %impl.elem0.b1a
// CHECK:STDOUT:       %specific_fn: <specific function> = specific_function %impl.elem0.b1a, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.4a2: <bound method> = bound_method %.25f, %specific_fn
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.4a2(%.25f)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.242: %tuple.type.f7a = tuple_value (%inst.splice_block) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl: %Destroy.WithSelf.SubobjectDestroy.type.dfc = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc6 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %i32 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %i32 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl: %Destroy.WithSelf.Op.type.ef0 = fn_decl @Destroy.WithSelf.Op.loc6 [concrete = constants.%Destroy.WithSelf.Op.403] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9d9 = ref_param_pattern [concrete = constants.%self.param_patt.6c0]
// CHECK:STDOUT:     %self.patt: %pattern_type.9d9 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.1ea]
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
// CHECK:STDOUT:   %.loc6_3.2: %tuple.type.f7a = initialize_action %x.ref, constants.%i32, %w.var, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc6_3.2, element0 [template]
// CHECK:STDOUT:   %.loc6_3.3: %i32 = splice_inst %tuple.elem0 [template = %.loc6_3.3 (constants.%.721)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc5_16.1 (%T.67db0b.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref %i32 = var_storage %w.var_patt
// CHECK:STDOUT:     %x.ref: @F.%T.loc5_16.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %.loc6_3.1: %i32 = splice_inst %tuple.elem0 [template = %.loc6_3.3 (constants.%.721)]
// CHECK:STDOUT:     assign %w.var, %.loc6_3.1
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %w: ref %i32 = wrapper_binding w, %w.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %w.patt: %pattern_type.9d9 = ref_binding_pattern w [concrete = constants.%w.patt]
// CHECK:STDOUT:       %w.var_patt: %pattern_type.9d9 = var_pattern %w.patt [concrete = constants.%w.var_patt]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %w.var, constants.%Destroy.WithSelf.SelfDestruct.db3
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%w.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc6(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc6(%self.param: ref %i32) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc6(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl(%self.param)
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT:   %x.param_patt.loc5_25.2 => constants.%x.param_patt.3bc
// CHECK:STDOUT:   %x.patt.loc5_25.2 => constants.%x.patt.0ff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_3.2 => constants.%tuple.242
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block
// CHECK:STDOUT:   %.loc6_3.3 => <not constant>
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
// CHECK:STDOUT:   %T.patt.3a0592.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T.67db0b.1: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %w.patt.f6d: %pattern_type.51d1c4.1 = ref_binding_pattern w [template]
// CHECK:STDOUT:   %w.var_patt.7fe: %pattern_type.51d1c4.1 = var_pattern %w.patt.f6d [template]
// CHECK:STDOUT:   %tuple.type.92e: type = tuple_type (%X, %X, %X) [concrete]
// CHECK:STDOUT:   %tuple.type.f39: type = tuple_type (<instruction>, <instruction>, <instruction>, <instruction>) [concrete]
// CHECK:STDOUT:   %.4e3: %X = splice_inst @F.%tuple.elem1 [template]
// CHECK:STDOUT:   %.c57: %X = splice_inst @F.%tuple.elem2 [template]
// CHECK:STDOUT:   %.ceb: %X = splice_inst @F.%tuple.elem3 [template]
// CHECK:STDOUT:   %.b00: %T.67db0b.1 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.869: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.c97: %Destroy.WithSelf.SelfDestruct.type.869 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.df0 [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2cb: type = facet_type <@ImplicitAs, @ImplicitAs(%Y)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @tuple.type.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.eca: type = pattern_type %tuple.type.92e [concrete]
// CHECK:STDOUT:   %pattern_type.7ab: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %tuple.type.as.ImplicitAs.impl.Convert.type: type = fn_type @tuple.type.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %tuple.type.as.ImplicitAs.impl.Convert: %tuple.type.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2cb = facet_value %tuple.type.92e, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c71: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Y, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %int_3: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.f68: type = class_type @Array, @Array(%X, %int_3) [concrete]
// CHECK:STDOUT:   %complete_type.8ad: <witness> = complete_type_witness %tuple.type.92e [concrete]
// CHECK:STDOUT:   %w.patt.689: %pattern_type.eca = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.e50: %pattern_type.eca = var_pattern %w.patt.689 [concrete]
// CHECK:STDOUT:   %inst.tuple_init: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f58: init %tuple.type.92e to %.7189bc.2 = tuple_init (@F.%Make.call.loc9_27, @F.%Make.call.loc9_35, @F.%Make.call.loc9_43)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.521: ref %X = splice_block %tuple.elem0 {
// CHECK:STDOUT:       %.7189bc.2: ref %tuple.type.92e = specific_inst @F.%w.var, @F(%tuple.type.92e)
// CHECK:STDOUT:       %tuple.elem0: ref %X = tuple_access %.7189bc.2, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem1: ref %X = tuple_access %.7189bc.2, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem2: ref %X = tuple_access %.7189bc.2, element2
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.b09: %pattern_type.eca = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.6d4: %pattern_type.eca = wrapper_binding_pattern self, %self.param_patt.b09 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc8af.1: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %int_1.5b8115.1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0.5c61a1.1: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.3: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.3, %Destroy.WithSelf.SubobjectDestroy.d01daf.3, %Destroy.WithSelf.SelfDestruct.db3fdb.3), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.51e: %Destroy.type = facet_value %tuple.type.92e, (%custom_witness.f8f19d.3) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.0b2: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.51e) [concrete]
// CHECK:STDOUT:   %.095: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.0b2, %Destroy.facet.51e [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.8f6: <bound method> = splice_block %bound_method.649 {
// CHECK:STDOUT:       %.7189bc.1: ref %tuple.type.92e = specific_inst @F.%w.var, @F(%tuple.type.92e)
// CHECK:STDOUT:       %impl.elem2.23d: %.095 = impl_witness_access %custom_witness.f8f19d.3, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.3]
// CHECK:STDOUT:       %bound_method.649: <bound method> = bound_method %.7189bc.1, %impl.elem2.23d
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.60b: init %empty_tuple.type = call %.8f6(%.7189bc.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %array_type.4c0: type = array_type %int_3, %X [concrete]
// CHECK:STDOUT:   %complete_type.3e8: <witness> = complete_type_witness %array_type.4c0 [concrete]
// CHECK:STDOUT:   %pattern_type.5e2: type = pattern_type %Array.f68 [concrete]
// CHECK:STDOUT:   %w.patt.979: %pattern_type.5e2 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.697: %pattern_type.5e2 = var_pattern %w.patt.979 [concrete]
// CHECK:STDOUT:   %inst.array_init: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.2eb: init %Array.f68 to %.11ccfd.2 = array_init (@F.%Make.call.loc9_27, @F.%Make.call.loc9_35, @F.%Make.call.loc9_43)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.08e: ref %X = splice_block %.48c {
// CHECK:STDOUT:       %.11ccfd.2: ref %Array.f68 = specific_inst @F.%w.var, @F(%Array.f68)
// CHECK:STDOUT:       %int_0.5c61a1.2: Core.IntLiteral = int_value 0 [concrete = %int_0.5c61a1.1]
// CHECK:STDOUT:       %.48c: ref %X = array_index %.11ccfd.2, %int_0.5c61a1.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.6e8: ref %X = splice_block %.2c0 {
// CHECK:STDOUT:       %int_1.5b8115.2: Core.IntLiteral = int_value 1 [concrete = %int_1.5b8115.1]
// CHECK:STDOUT:       %.2c0: ref %X = array_index %.11ccfd.2, %int_1.5b8115.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.631: ref %X = splice_block %.468 {
// CHECK:STDOUT:       %int_2.ecc8af.2: Core.IntLiteral = int_value 2 [concrete = %int_2.ecc8af.1]
// CHECK:STDOUT:       %.468: ref %X = array_index %.11ccfd.2, %int_2.ecc8af.2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple.1ed7a0.2: %tuple.type.f39 = tuple_value (%inst.array_init, %inst.splice_block.426619.3, %inst.splice_block.426619.4, %inst.splice_block.426619.5) [concrete]
// CHECK:STDOUT:   %self.param_patt.910: %pattern_type.5e2 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.74a: %pattern_type.5e2 = wrapper_binding_pattern self, %self.param_patt.910 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.4: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.4, %Destroy.WithSelf.SubobjectDestroy.d01daf.4, %Destroy.WithSelf.SelfDestruct.db3fdb.4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2b9: %Destroy.type = facet_value %Array.f68, (%custom_witness.f8f19d.4) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.bd6: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.2b9) [concrete]
// CHECK:STDOUT:   %.de1: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.bd6, %Destroy.facet.2b9 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.0f9: <bound method> = splice_block %bound_method.5e9 {
// CHECK:STDOUT:       %.11ccfd.1: ref %Array.f68 = specific_inst @F.%w.var, @F(%Array.f68)
// CHECK:STDOUT:       %impl.elem2.74d: %.de1 = impl_witness_access %custom_witness.f8f19d.4, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.4]
// CHECK:STDOUT:       %bound_method.5e9: <bound method> = bound_method %.11ccfd.1, %impl.elem2.74d
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.972: init %empty_tuple.type = call %.0f9(%.11ccfd.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %w.patt.22a: %pattern_type.7ab = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.565: %pattern_type.7ab = var_pattern %w.patt.22a [concrete]
// CHECK:STDOUT:   %.321: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.c71, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.897: init %Y = splice_block %tuple.type.as.ImplicitAs.impl.Convert.call {
// CHECK:STDOUT:       %impl.elem0.4aa: %.321 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %tuple.type.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.454: <bound method> = bound_method @F.%.loc9_44, %impl.elem0.4aa
// CHECK:STDOUT:       %.12a9d9.2: ref %Y = specific_inst @F.%w.var, @F(%Y)
// CHECK:STDOUT:       %.71d: ref %X = temporary @F.%.loc9_3.1, @F.%Make.call.loc9_27
// CHECK:STDOUT:       %.173: %X = acquire_value %.71d
// CHECK:STDOUT:       %.61c: ref %X = temporary @F.%.loc9_3.2, @F.%Make.call.loc9_35
// CHECK:STDOUT:       %.f2a: %X = acquire_value %.61c
// CHECK:STDOUT:       %.55f: ref %X = temporary @F.%.loc9_3.3, @F.%Make.call.loc9_43
// CHECK:STDOUT:       %.eee: %X = acquire_value %.55f
// CHECK:STDOUT:       %tuple.9aa: %tuple.type.92e = tuple_value (%.173, %.f2a, %.eee)
// CHECK:STDOUT:       %.1f9: %tuple.type.92e = converted @F.%.loc9_44, %tuple.9aa
// CHECK:STDOUT:       %tuple.type.as.ImplicitAs.impl.Convert.call: init %Y to %.12a9d9.2 = call %bound_method.454(%.1f9)
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.062: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.09f) [concrete]
// CHECK:STDOUT:   %.e55: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.062, %Destroy.facet.09f [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c75: <bound method> = splice_block %bound_method.d1a {
// CHECK:STDOUT:       %.12a9d9.1: ref %Y = specific_inst @F.%w.var, @F(%Y)
// CHECK:STDOUT:       %impl.elem2.f87: %.e55 = impl_witness_access %custom_witness.f8f19d.5, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.5]
// CHECK:STDOUT:       %bound_method.d1a: <bound method> = bound_method %.12a9d9.1, %impl.elem2.f87
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.38e: init %empty_tuple.type = call %.c75(%.12a9d9.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.df0: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.869) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.c97)]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5e2 = ref_param_pattern [concrete = constants.%self.param_patt.910]
// CHECK:STDOUT:     %self.patt: %pattern_type.5e2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.74a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.f68 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.f68 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc9_3.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5e2 = ref_param_pattern [concrete = constants.%self.param_patt.910]
// CHECK:STDOUT:     %self.patt: %pattern_type.5e2 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.74a]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %Array.f68 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %Array.f68 = wrapper_binding self, %self.param
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
// CHECK:STDOUT:     %T.patt.loc8_16.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_18.1: type = splice_block %.loc8_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc8_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_16.2: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc8_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_16.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc8_16.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:   %T.loc8_16.1: type = symbolic_binding T, 0, template [template = %T.loc8_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc8_16.1 [template = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc8_16.1 [template = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %w.patt.loc9_15.2: @F.%pattern_type (%pattern_type.51d1c4.1) = ref_binding_pattern w [template = %w.patt.loc9_15.2 (constants.%w.patt.f6d)]
// CHECK:STDOUT:   %w.var_patt.loc9_3.2: @F.%pattern_type (%pattern_type.51d1c4.1) = var_pattern %w.patt.loc9_15.2 [template = %w.var_patt.loc9_3.2 (constants.%w.var_patt.7fe)]
// CHECK:STDOUT:   %.loc9_3.9: %tuple.type.f39 = initialize_action %.loc9_44, %T.loc8_16.1, %w.var, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc9_3.9, element0 [template]
// CHECK:STDOUT:   %tuple.elem1: <instruction> = tuple_access %.loc9_3.9, element1 [template]
// CHECK:STDOUT:   %.loc9_3.10: %X = splice_inst %tuple.elem1 [template = %.loc9_3.10 (constants.%.4e3)]
// CHECK:STDOUT:   %tuple.elem2: <instruction> = tuple_access %.loc9_3.9, element2 [template]
// CHECK:STDOUT:   %.loc9_3.11: %X = splice_inst %tuple.elem2 [template = %.loc9_3.11 (constants.%.c57)]
// CHECK:STDOUT:   %tuple.elem3: <instruction> = tuple_access %.loc9_3.9, element3 [template]
// CHECK:STDOUT:   %.loc9_3.12: %X = splice_inst %tuple.elem3 [template = %.loc9_3.12 (constants.%.ceb)]
// CHECK:STDOUT:   %.loc9_3.13: @F.%T.loc8_16.1 (%T.67db0b.1) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.b00)]
// CHECK:STDOUT:   %.loc9_3.14: <instruction> = compound_member_access_action %w.var, constants.%assoc2 [template]
// CHECK:STDOUT:   %.loc9_3.15: type = type_of_inst %.loc9_3.14 [template = %.loc9_3.15 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc9_3.16: @F.%.loc9_3.15 (%.8cb) = splice_inst %.loc9_3.14 [template = %.loc9_3.16 (constants.%.c23)]
// CHECK:STDOUT:   %.loc9_3.17: <instruction> = call_action (%.loc9_3.6), true [template]
// CHECK:STDOUT:   %.loc9_3.18: type = type_of_inst %.loc9_3.17 [template = %.loc9_3.18 (constants.%.57c)]
// CHECK:STDOUT:   %.loc9_3.19: @F.%.loc9_3.18 (%.57c) = splice_inst %.loc9_3.17 [template = %.loc9_3.19 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %w.var: ref @F.%T.loc8_16.1 (%T.67db0b.1) = var_storage %w.var_patt.loc9_3.1
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
// CHECK:STDOUT:     %.loc9_3.4: @F.%T.loc8_16.1 (%T.67db0b.1) = splice_inst %tuple.elem0 [template = %.loc9_3.13 (constants.%.b00)]
// CHECK:STDOUT:     assign %w.var, %.loc9_3.4
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_16.2 [template = %T.loc8_16.1 (constants.%T.67db0b.1)]
// CHECK:STDOUT:     %w: ref @F.%T.loc8_16.1 (%T.67db0b.1) = wrapper_binding w, %w.var
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %X) = "no_op";
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
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc8af.1]
// CHECK:STDOUT:   %tuple.elem2: ref %X = tuple_access %self.param, element2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %tuple.elem2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%tuple.elem2)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8115.1]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c61a1.1]
// CHECK:STDOUT:   %tuple.elem0: ref %X = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3(%tuple.elem0)
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.4(%self.param: ref %Array.f68) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc8af.1]
// CHECK:STDOUT:   %.loc9_3.1: ref %X = array_index %self.param, %int_2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8115.1]
// CHECK:STDOUT:   %.loc9_3.2: ref %X = array_index %self.param, %int_1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%.loc9_3.2)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c61a1.1]
// CHECK:STDOUT:   %.loc9_3.3: ref %X = array_index %self.param, %int_0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3: <bound method> = bound_method %.loc9_3.3, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.3: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.3(%.loc9_3.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_3.4(%self.param: ref %Array.f68) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_3.4(%self.param: ref %Array.f68) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.5(%self.param: ref %Y) = "no_op";
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
// CHECK:STDOUT: specific @F(constants.%T.67db0b.1) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%T.67db0b.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%tuple.type.92e) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.3a0592.1
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
// CHECK:STDOUT:   %.loc9_3.10 => <not constant>
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.tuple_access.426619.1
// CHECK:STDOUT:   %.loc9_3.11 => <not constant>
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.tuple_access.426619.2
// CHECK:STDOUT:   %.loc9_3.12 => <not constant>
// CHECK:STDOUT:   %.loc9_3.13 => <not constant>
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => <not constant>
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.1
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Array.f68) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.3a0592.1
// CHECK:STDOUT:   %T.loc8_16.1 => constants.%Array.f68
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3e8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.5e2
// CHECK:STDOUT:   %w.patt.loc9_15.2 => constants.%w.patt.979
// CHECK:STDOUT:   %w.var_patt.loc9_3.2 => constants.%w.var_patt.697
// CHECK:STDOUT:   %.loc9_3.9 => constants.%tuple.1ed7a0.2
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.array_init
// CHECK:STDOUT:   %tuple.elem1 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc9_3.10 => <not constant>
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc9_3.11 => <not constant>
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc9_3.12 => <not constant>
// CHECK:STDOUT:   %.loc9_3.13 => <not constant>
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => <not constant>
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.2
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Y) {
// CHECK:STDOUT:   %T.patt.loc8_16.2 => constants.%T.patt.3a0592.1
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
// CHECK:STDOUT:   %.loc9_3.10 => <not constant>
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.temporary_storage.426619.2
// CHECK:STDOUT:   %.loc9_3.11 => <not constant>
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.temporary_storage.426619.3
// CHECK:STDOUT:   %.loc9_3.12 => <not constant>
// CHECK:STDOUT:   %.loc9_3.13 => <not constant>
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.8
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => <not constant>
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.3
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => <not constant>
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.869: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.c97: %Destroy.WithSelf.SelfDestruct.type.869 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.df0 [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc9_3.14 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc9_3.17 [template]
// CHECK:STDOUT:   %Y: type = class_type @Y [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.2cb: type = facet_type <@ImplicitAs, @ImplicitAs(%Y)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @X.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.7ab: type = pattern_type %Y [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert.type: type = fn_type @X.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %X.as.ImplicitAs.impl.Convert: %X.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.2cb = facet_value %X, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.9eb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%Y, %ImplicitAs.facet) [concrete]
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
// CHECK:STDOUT:     %.631: init %tuple.type.4c2 = splice_block %.66c {
// CHECK:STDOUT:       %.04a: init %struct_type.a.b.f38 to %tuple.elem0.16b = struct_init (@F.%Make.call.loc9_33, @F.%Make.call.loc9_46)
// CHECK:STDOUT:       %.2cf: init %struct_type.a.b.f38 = converted @F.%.loc9_47, %.04a
// CHECK:STDOUT:       %.66c: init %tuple.type.4c2 to %.224b0e.2 = tuple_init (%.2cf, @F.%Make.call.loc9_55)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.d5e: ref %X = splice_block %.ae2 {
// CHECK:STDOUT:       %.224b0e.2: ref %tuple.type.4c2 = specific_inst @F.%w.var, @F(%tuple.type.4c2)
// CHECK:STDOUT:       %tuple.elem0.16b: ref %struct_type.a.b.f38 = tuple_access %.224b0e.2, element0
// CHECK:STDOUT:       %.ae2: ref %X = struct_access %tuple.elem0.16b, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.struct_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ffd: ref %X = struct_access %tuple.elem0.16b, element1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.tuple_access: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %tuple.elem1.b82: ref %X = tuple_access %.224b0e.2, element1
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.3: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.3: %Destroy.WithSelf.SelfDestruct.type.fbceb5.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.165: %pattern_type.9a0 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c78: %pattern_type.9a0 = wrapper_binding_pattern self, %self.param_patt.165 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.f8f19d.4: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.4, %Destroy.WithSelf.SubobjectDestroy.d01daf.4, %Destroy.WithSelf.SelfDestruct.db3fdb.4), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.e65: %Destroy.type = facet_value %tuple.type.4c2, (%custom_witness.f8f19d.4) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.839: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.e65) [concrete]
// CHECK:STDOUT:   %.81f: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.839, %Destroy.facet.e65 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.1f2: <bound method> = splice_block %bound_method.a78 {
// CHECK:STDOUT:       %.224b0e.1: ref %tuple.type.4c2 = specific_inst @F.%w.var, @F(%tuple.type.4c2)
// CHECK:STDOUT:       %impl.elem2.006: %.81f = impl_witness_access %custom_witness.f8f19d.4, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.4]
// CHECK:STDOUT:       %bound_method.a78: <bound method> = bound_method %.224b0e.1, %impl.elem2.006
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.2c9: init %empty_tuple.type = call %.1f2(%.224b0e.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.8ea: <witness> = complete_type_witness %tuple.type.2f4 [concrete]
// CHECK:STDOUT:   %pattern_type.2cd: type = pattern_type %tuple.type.2f4 [concrete]
// CHECK:STDOUT:   %w.patt.e8c: %pattern_type.2cd = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.61a: %pattern_type.2cd = var_pattern %w.patt.e8c [concrete]
// CHECK:STDOUT:   %.6a7: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.9eb, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.270: init %tuple.type.2f4 = splice_block %.7fa {
// CHECK:STDOUT:       %impl.elem0.6f6eb7.3: %.6a7 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.1a0: <bound method> = bound_method @F.%Make.call.loc9_33, %impl.elem0.6f6eb7.3
// CHECK:STDOUT:       %.f80: ref %Y = splice_block %.ce4 {
// CHECK:STDOUT:         %.021893.2: ref %tuple.type.2f4 = specific_inst @F.%w.var, @F(%tuple.type.2f4)
// CHECK:STDOUT:         %tuple.elem0.c3c: ref %struct_type.a.b.095 = tuple_access %.021893.2, element0
// CHECK:STDOUT:         %.ce4: ref %Y = struct_access %tuple.elem0.c3c, element0
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %.71d: ref %X = temporary @F.%.loc9_3.1, @F.%Make.call.loc9_33
// CHECK:STDOUT:       %.173: %X = acquire_value %.71d
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.495: init %Y to %.f80 = call %bound_method.1a0(%.173)
// CHECK:STDOUT:       %.59c: init %Y = converted @F.%Make.call.loc9_33, %X.as.ImplicitAs.impl.Convert.call.495
// CHECK:STDOUT:       %impl.elem0.6f6eb7.4: %.6a7 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.b4c909.2: <bound method> = bound_method @F.%Make.call.loc9_46, %impl.elem0.6f6eb7.4
// CHECK:STDOUT:       %.0ae: ref %Y = struct_access %tuple.elem0.c3c, element1
// CHECK:STDOUT:       %.61c809.2: ref %X = temporary @F.%.loc9_3.2, @F.%Make.call.loc9_46
// CHECK:STDOUT:       %.f2a65d.2: %X = acquire_value %.61c809.2
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.879: init %Y to %.0ae = call %bound_method.b4c909.2(%.f2a65d.2)
// CHECK:STDOUT:       %.d94: init %Y = converted @F.%Make.call.loc9_46, %X.as.ImplicitAs.impl.Convert.call.879
// CHECK:STDOUT:       %.7de: init %struct_type.a.b.095 to %tuple.elem0.c3c = struct_init (%.59c, %.d94)
// CHECK:STDOUT:       %.e70: init %struct_type.a.b.095 = converted @F.%.loc9_47, %.7de
// CHECK:STDOUT:       %impl.elem0.6f6eb7.5: %.6a7 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.83e768.2: <bound method> = bound_method @F.%Make.call.loc9_55, %impl.elem0.6f6eb7.5
// CHECK:STDOUT:       %tuple.elem1.1e9: ref %Y = tuple_access %.021893.2, element1
// CHECK:STDOUT:       %.55fb1a.2: ref %X = temporary @F.%.loc9_3.3, @F.%Make.call.loc9_55
// CHECK:STDOUT:       %.eeedf6.2: %X = acquire_value %.55fb1a.2
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.602: init %Y to %tuple.elem1.1e9 = call %bound_method.83e768.2(%.eeedf6.2)
// CHECK:STDOUT:       %.725: init %Y = converted @F.%Make.call.loc9_55, %X.as.ImplicitAs.impl.Convert.call.602
// CHECK:STDOUT:       %.7fa: init %tuple.type.2f4 to %.021893.2 = tuple_init (%.e70, %.725)
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.6: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.6 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.6: %Destroy.WithSelf.SelfDestruct.type.fbceb5.6 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.18f: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.585) [concrete]
// CHECK:STDOUT:   %.2b6: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.18f, %Destroy.facet.585 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.c0f: <bound method> = splice_block %bound_method.349 {
// CHECK:STDOUT:       %.021893.1: ref %tuple.type.2f4 = specific_inst @F.%w.var, @F(%tuple.type.2f4)
// CHECK:STDOUT:       %impl.elem2.b50: %.2b6 = impl_witness_access %custom_witness.f8f19d.7, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.7]
// CHECK:STDOUT:       %bound_method.349: <bound method> = bound_method %.021893.1, %impl.elem2.b50
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.7f9: init %empty_tuple.type = call %.c0f(%.021893.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type.561: <witness> = complete_type_witness %tuple.type.bb4 [concrete]
// CHECK:STDOUT:   %pattern_type.8b0: type = pattern_type %tuple.type.bb4 [concrete]
// CHECK:STDOUT:   %w.patt.d74: %pattern_type.8b0 = ref_binding_pattern w [concrete]
// CHECK:STDOUT:   %w.var_patt.e1e: %pattern_type.8b0 = var_pattern %w.patt.d74 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.383: init %tuple.type.bb4 = splice_block %.74e {
// CHECK:STDOUT:       %impl.elem0.6f6eb7.1: %.6a7 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.b4c909.1: <bound method> = bound_method @F.%Make.call.loc9_46, %impl.elem0.6f6eb7.1
// CHECK:STDOUT:       %.94f: ref %Y = class_element_access %tuple.elem0.297, element1
// CHECK:STDOUT:       %.61c809.1: ref %X = temporary @F.%.loc9_3.2, @F.%Make.call.loc9_46
// CHECK:STDOUT:       %.f2a65d.1: %X = acquire_value %.61c809.1
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.9d2: init %Y to %.94f = call %bound_method.b4c909.1(%.f2a65d.1)
// CHECK:STDOUT:       %.644: init %Y = converted @F.%Make.call.loc9_46, %X.as.ImplicitAs.impl.Convert.call.9d2
// CHECK:STDOUT:       %.b4b: init %Z to %tuple.elem0.297 = class_init (@F.%Make.call.loc9_33, %.644)
// CHECK:STDOUT:       %.cac: init %Z = converted @F.%.loc9_47, %.b4b
// CHECK:STDOUT:       %impl.elem0.6f6eb7.2: %.6a7 = impl_witness_access %ImplicitAs.impl_witness, element0 [concrete = %X.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:       %bound_method.83e768.1: <bound method> = bound_method @F.%Make.call.loc9_55, %impl.elem0.6f6eb7.2
// CHECK:STDOUT:       %tuple.elem1.76f: ref %Y = tuple_access %.9597f3.2, element1
// CHECK:STDOUT:       %.55fb1a.1: ref %X = temporary @F.%.loc9_3.3, @F.%Make.call.loc9_55
// CHECK:STDOUT:       %.eeedf6.1: %X = acquire_value %.55fb1a.1
// CHECK:STDOUT:       %X.as.ImplicitAs.impl.Convert.call.57a: init %Y to %tuple.elem1.76f = call %bound_method.83e768.1(%.eeedf6.1)
// CHECK:STDOUT:       %.265: init %Y = converted @F.%Make.call.loc9_55, %X.as.ImplicitAs.impl.Convert.call.57a
// CHECK:STDOUT:       %.74e: init %tuple.type.bb4 to %.9597f3.2 = tuple_init (%.cac, %.265)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.7: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.1d4: ref %X = splice_block %.014 {
// CHECK:STDOUT:       %.9597f3.2: ref %tuple.type.bb4 = specific_inst @F.%w.var, @F(%tuple.type.bb4)
// CHECK:STDOUT:       %tuple.elem0.297: ref %Z = tuple_access %.9597f3.2, element0
// CHECK:STDOUT:       %.014: ref %X = class_element_access %tuple.elem0.297, element0
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.9: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_3.9 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.9: %Destroy.WithSelf.SelfDestruct.type.fbceb5.9 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.44e: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.128) [concrete]
// CHECK:STDOUT:   %.560: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.44e, %Destroy.facet.128 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.8: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.9e3: <bound method> = splice_block %bound_method.d0a {
// CHECK:STDOUT:       %.9597f3.1: ref %tuple.type.bb4 = specific_inst @F.%w.var, @F(%tuple.type.bb4)
// CHECK:STDOUT:       %impl.elem2.21a: %.560 = impl_witness_access %custom_witness.f8f19d.10, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.10]
// CHECK:STDOUT:       %bound_method.d0a: <bound method> = bound_method %.9597f3.1, %impl.elem2.21a
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call.cbc: init %empty_tuple.type = call %.9e3(%.9597f3.1)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.df0: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.869) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.c97)]
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
// CHECK:STDOUT:   %.loc9_3.9: %tuple.type.f39 = initialize_action %.loc9_56, %T.loc8_16.1, %w.var, none, false [template]
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.2(%self.param: ref %X) = "no_op";
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
// CHECK:STDOUT:   %.loc9_3.1: ref %X = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %X = struct_access %self.param, element0
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
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %X = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %struct_type.a.b.f38 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.3
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%tuple.elem0)
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
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_3.5(%self.param: ref %Y) = "no_op";
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
// CHECK:STDOUT:   %.loc9_3.1: ref %Y = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %Y = struct_access %self.param, element0
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
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %Y = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %struct_type.a.b.095 = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.6
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%tuple.elem0)
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
// CHECK:STDOUT:   %.loc9_3.1: ref %Y = struct_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %.loc9_3.2: ref %X = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
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
// CHECK:STDOUT:   %b.ref: %Z.elem.5d0 = name_ref b, @Z.%field_decl.loc21 [concrete = @Z.%field_decl.loc21]
// CHECK:STDOUT:   %.loc9_3.1: ref %Y = class_element_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %.loc9_3.1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%.loc9_3.1)
// CHECK:STDOUT:   %a.ref: %Z.elem.d23 = name_ref a, @Z.%field_decl.loc20 [concrete = @Z.%field_decl.loc20]
// CHECK:STDOUT:   %.loc9_3.2: ref %X = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %.loc9_3.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
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
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %tuple.elem1: ref %Y = tuple_access %self.param, element1
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1: <bound method> = bound_method %tuple.elem1, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.5
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.1: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.1(%tuple.elem1)
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0: ref %Z = tuple_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2: <bound method> = bound_method %tuple.elem0, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.9
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call.loc9_3.2: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound.loc9_3.2(%tuple.elem0)
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
// CHECK:STDOUT:   %.loc9_3.10 => <not constant>
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.struct_access
// CHECK:STDOUT:   %.loc9_3.11 => <not constant>
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.tuple_access
// CHECK:STDOUT:   %.loc9_3.12 => <not constant>
// CHECK:STDOUT:   %.loc9_3.13 => <not constant>
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => <not constant>
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.1
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => <not constant>
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
// CHECK:STDOUT:   %.loc9_3.10 => <not constant>
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.temporary_storage.426619.2
// CHECK:STDOUT:   %.loc9_3.11 => <not constant>
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.temporary_storage.426619.3
// CHECK:STDOUT:   %.loc9_3.12 => <not constant>
// CHECK:STDOUT:   %.loc9_3.13 => <not constant>
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => <not constant>
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.2
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => <not constant>
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
// CHECK:STDOUT:   %.loc9_3.10 => <not constant>
// CHECK:STDOUT:   %tuple.elem2 => constants.%inst.temporary_storage.426619.4
// CHECK:STDOUT:   %.loc9_3.11 => <not constant>
// CHECK:STDOUT:   %tuple.elem3 => constants.%inst.temporary_storage.426619.5
// CHECK:STDOUT:   %.loc9_3.12 => <not constant>
// CHECK:STDOUT:   %.loc9_3.13 => <not constant>
// CHECK:STDOUT:   %.loc9_3.14 => constants.%inst.splice_block.426619.8
// CHECK:STDOUT:   %.loc9_3.15 => <bound method>
// CHECK:STDOUT:   %.loc9_3.16 => <not constant>
// CHECK:STDOUT:   %.loc9_3.17 => constants.%inst.call.426619.3
// CHECK:STDOUT:   %.loc9_3.18 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc9_3.19 => <not constant>
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
// CHECK:STDOUT:   %.55c: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.13f: %pattern_type.9d9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a79: %pattern_type.9d9 = return_slot_pattern %return.param_patt.13f, %i32 [concrete]
// CHECK:STDOUT:   %AsI32.type: type = fn_type @AsI32 [concrete]
// CHECK:STDOUT:   %AsI32: %AsI32.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67db0b.1 [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %.796: %i32 = splice_inst @AsI32.%.loc6_12.2 [template]
// CHECK:STDOUT:   %.e20: %i32 = splice_inst @AsI32.%.loc6_18.2 [template]
// CHECK:STDOUT:   %x.param_patt.3bc: %pattern_type.9d9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0ff: %pattern_type.9d9 = wrapper_binding_pattern x, %x.param_patt.3bc [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %As.type.d76: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.1ed: <witness> = impl_witness @C.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert.type: type = fn_type @C.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert: %C.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d76 = facet_value %C, (%As.impl_witness.1ed) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.a91: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.26a: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.d84: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cca: %Int.as.Copy.impl.Op.type.d84 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0ff: <witness> = impl_witness imports.%Copy.impl_witness_table.f74, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a63: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.617: %Int.as.Copy.impl.Op.type.a63 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0ff) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.bc1: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.481: type = fn_type_with_self_type %Copy.WithSelf.Op.type.bc1, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.617, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %inst.specific_inst: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.79b: %i32 = specific_inst @AsI32.%x.ref, @AsI32(%i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.09a: init %i32 = splice_block %Int.as.Copy.impl.Op.call.3a9 {
// CHECK:STDOUT:       %impl.elem0.b1ab63.2: %.481 = impl_witness_access %Copy.impl_witness.0ff, element0 [concrete = %Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:       %bound_method.9e8: <bound method> = bound_method %.79b, %impl.elem0.b1ab63.2
// CHECK:STDOUT:       %specific_fn.57cbf3.2: <specific function> = specific_function %impl.elem0.b1ab63.2, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.3a6: <bound method> = bound_method %.79b, %specific_fn.57cbf3.2
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call.3a9: init %i32 = call %bound_method.3a6(%.79b)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.695: type = fn_type_with_self_type %As.WithSelf.Convert.type.a91, %As.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.388: %i32 = splice_block %.9a8 {
// CHECK:STDOUT:       %.61f: %C = specific_inst @AsI32.%x.ref, @AsI32(%C)
// CHECK:STDOUT:       %impl.elem0.e7b: %.695 = impl_witness_access %As.impl_witness.1ed, element0 [concrete = %C.as.As.impl.Convert]
// CHECK:STDOUT:       %bound_method.ff4: <bound method> = bound_method %.61f, %impl.elem0.e7b
// CHECK:STDOUT:       %C.as.As.impl.Convert.call: init %i32 = call %bound_method.ff4(%.61f)
// CHECK:STDOUT:       %.9a8: %i32 = value_of_initializer %C.as.As.impl.Convert.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.dfa: init %i32 = splice_block %Int.as.Copy.impl.Op.call.eb3 {
// CHECK:STDOUT:       %impl.elem0.b1ab63.1: %.481 = impl_witness_access %Copy.impl_witness.0ff, element0 [concrete = %Int.as.Copy.impl.Op.617]
// CHECK:STDOUT:       %bound_method.6eb: <bound method> = bound_method %.388, %impl.elem0.b1ab63.1
// CHECK:STDOUT:       %specific_fn.57cbf3.1: <specific function> = specific_function %impl.elem0.b1ab63.1, @Int.as.Copy.impl.Op(%int_32) [concrete = %Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:       %bound_method.4a9: <bound method> = bound_method %.388, %specific_fn.57cbf3.1
// CHECK:STDOUT:       %Int.as.Copy.impl.Op.call.eb3: init %i32 = call %bound_method.4a9(%.388)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.dfa: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.d84) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.cca)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f74 = impl_witness_table (%Core.import_ref.dfa), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %AsI32.decl: %AsI32.type = fn_decl @AsI32 [concrete = constants.%AsI32] {
// CHECK:STDOUT:     %T.patt.loc5_20.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc5_20.2 (constants.%T.patt.3a0592.1)]
// CHECK:STDOUT:     %x.param_patt.loc5_29.1: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = value_param_pattern [template = %x.param_patt.loc5_29.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc5_29.1: @AsI32.%pattern_type (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc5_29.1 [template = %x.patt.loc5_29.2 (constants.%x.patt.260)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9d9 = out_param_pattern [concrete = constants.%return.param_patt.13f]
// CHECK:STDOUT:     %return.patt: %pattern_type.9d9 = return_slot_pattern %return.param_patt, %i32.loc5 [concrete = constants.%return.patt.a79]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc5_37: Core.Form = init_form %i32.loc5 [concrete = constants.%.55c]
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
// CHECK:STDOUT:   %.loc6_12.3: %i32 = splice_inst %.loc6_12.2 [template = %.loc6_12.3 (constants.%.796)]
// CHECK:STDOUT:   %.loc6_18.2: <instruction> = convert_to_category_action %.loc6_12.1, element10 [template]
// CHECK:STDOUT:   %.loc6_18.3: %i32 = splice_inst %.loc6_18.2 [template = %.loc6_18.3 (constants.%.e20)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @AsI32.%T.loc5_20.1 (%T.67db0b.1)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @AsI32.%T.loc5_20.1 (%T.67db0b.1) = name_ref x, %x
// CHECK:STDOUT:     %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_12.1: %i32 = splice_inst %.loc6_12.2 [template = %.loc6_12.3 (constants.%.796)]
// CHECK:STDOUT:     %.loc6_18.1: %i32 = splice_inst %.loc6_18.2 [template = %.loc6_18.3 (constants.%.e20)]
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
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9d9
// CHECK:STDOUT:   %x.param_patt.loc5_29.2 => constants.%x.param_patt.3bc
// CHECK:STDOUT:   %x.patt.loc5_29.2 => constants.%x.patt.0ff
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %.loc6_12.2 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_12.3 => <not constant>
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_18.3 => <not constant>
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.26a
// CHECK:STDOUT:   %.loc6_12.2 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_12.3 => <not constant>
// CHECK:STDOUT:   %.loc6_18.2 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_18.3 => <not constant>
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
// CHECK:STDOUT:   %pattern_type.9d9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %x.param_patt.3bc: %pattern_type.9d9 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.0ff: %pattern_type.9d9 = wrapper_binding_pattern x, %x.param_patt.3bc [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %As.type.d76: type = facet_type <@As, @As(%i32)> [concrete]
// CHECK:STDOUT:   %As.impl_witness.1ed: <witness> = impl_witness @C.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert.type: type = fn_type @C.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %C.as.As.impl.Convert: %C.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.d76 = facet_value %C, (%As.impl_witness.1ed) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.a91: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32, %As.facet) [concrete]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.26a: <witness> = complete_type_witness %struct_type.n [concrete]
// CHECK:STDOUT:   %x.param_patt.0f9: %pattern_type.98b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.953: %pattern_type.98b = wrapper_binding_pattern x, %x.param_patt.0f9 [concrete]
// CHECK:STDOUT:   %_.patt.6a3: %pattern_type.9d9 = value_binding_pattern _ [concrete]
// CHECK:STDOUT:   %inst.specific_inst: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.357: %i32 = specific_inst @AsType.%x.ref, @AsType(%i32, %i32)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.be9: %i32 = splice_block %.357 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.695: type = fn_type_with_self_type %As.WithSelf.Convert.type.a91, %As.facet [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f8b: %i32 = splice_block %.f89 {
// CHECK:STDOUT:       %.c96: %C = specific_inst @AsType.%x.ref, @AsType(%C, %i32)
// CHECK:STDOUT:       %impl.elem0.e7b: %.695 = impl_witness_access %As.impl_witness.1ed, element0 [concrete = %C.as.As.impl.Convert]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.c96, %impl.elem0.e7b
// CHECK:STDOUT:       %C.as.As.impl.Convert.call: init %i32 = call %bound_method(%.c96)
// CHECK:STDOUT:       %.f89: %i32 = value_of_initializer %C.as.As.impl.Convert.call
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.905: %i32 = splice_block %.f8b {}
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
// CHECK:STDOUT:   %pattern_type.loc5 => constants.%pattern_type.9d9
// CHECK:STDOUT:   %x.param_patt.loc5_30.2 => constants.%x.param_patt.3bc
// CHECK:STDOUT:   %x.patt.loc5_30.2 => constants.%x.patt.0ff
// CHECK:STDOUT:   %U.patt.loc5_45.2 => constants.%U.patt.cb8
// CHECK:STDOUT:   %U.loc5_45.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5 => constants.%complete_type.f8a
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.9d9
// CHECK:STDOUT:   %_.patt.loc6_8.2 => constants.%_.patt.6a3
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.specific_inst
// CHECK:STDOUT:   %.loc6_16.4 => <not constant>
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc6_16.6 => <not constant>
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
// CHECK:STDOUT:   %require_complete.loc5 => constants.%complete_type.26a
// CHECK:STDOUT:   %require_complete.loc6 => constants.%complete_type.f8a
// CHECK:STDOUT:   %pattern_type.loc6 => constants.%pattern_type.9d9
// CHECK:STDOUT:   %_.patt.loc6_8.2 => constants.%_.patt.6a3
// CHECK:STDOUT:   %.loc6_16.3 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc6_16.4 => <not constant>
// CHECK:STDOUT:   %.loc6_16.5 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc6_16.6 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- splice_inst_storage_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.MakeB.type: type = fn_type @B.MakeB [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.MakeB: %B.MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A.98c: type = class_type @A, @A(%T) [template]
// CHECK:STDOUT:   %.972: type = type_of_inst @A.MakeA.%.loc12_20.1 [template]
// CHECK:STDOUT:   %.a5e: %.972 = splice_inst @A.MakeA.%.loc12_20.1 [template]
// CHECK:STDOUT:   %.5372ba.1: type = splice_inst @A.MakeA.%.loc12_20.4 [template]
// CHECK:STDOUT:   %.cd79f1.1: Core.Form = init_form %.5372ba.1 [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.1: type = pattern_type %.5372ba.1 [template]
// CHECK:STDOUT:   %return.param_patt.2e9ac5.1: %pattern_type.fcb8f3.1 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7fc6a1.1: %pattern_type.fcb8f3.1 = return_slot_pattern %return.param_patt.2e9ac5.1, %.5372ba.1 [template]
// CHECK:STDOUT:   %.293: <instruction> = call_action (%A.generic, %T), false [template]
// CHECK:STDOUT:   %.3aa: type = type_of_inst %.293 [template]
// CHECK:STDOUT:   %.f2d: %.3aa = splice_inst %.293 [template]
// CHECK:STDOUT:   %.c8c: <instruction> = convert_to_value_action %.f2d, type [template]
// CHECK:STDOUT:   %.5372ba.2: type = splice_inst %.c8c [template]
// CHECK:STDOUT:   %.cd79f1.2: Core.Form = init_form %.5372ba.2 [template]
// CHECK:STDOUT:   %pattern_type.fcb8f3.2: type = pattern_type %.5372ba.2 [template]
// CHECK:STDOUT:   %return.param_patt.2e9ac5.2: %pattern_type.fcb8f3.2 = out_param_pattern [template]
// CHECK:STDOUT:   %return.patt.7fc6a1.2: %pattern_type.fcb8f3.2 = return_slot_pattern %return.param_patt.2e9ac5.2, %.5372ba.2 [template]
// CHECK:STDOUT:   %A.MakeA.type.27a: type = fn_type @A.MakeA, @A(%T) [template]
// CHECK:STDOUT:   %A.MakeA.f73: %A.MakeA.type.27a = struct_value () [template]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [template]
// CHECK:STDOUT:   %A.elem.754: type = unbound_element_type %A.98c, %T [template]
// CHECK:STDOUT:   %struct_type.field.686: type = struct_type {.field: %T} [template]
// CHECK:STDOUT:   %complete_type.725: <witness> = complete_type_witness %struct_type.field.686 [template]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %.5372ba.1 [template]
// CHECK:STDOUT:   %.a08: type = type_of_inst @A.MakeA.%.loc13_23.3 [template]
// CHECK:STDOUT:   %.5c2: %.a08 = splice_inst @A.MakeA.%.loc13_23.3 [template]
// CHECK:STDOUT:   %.308: type = type_of_inst @A.MakeA.%.loc13_30.3 [template]
// CHECK:STDOUT:   %.151: %.308 = splice_inst @A.MakeA.%.loc13_30.3 [template]
// CHECK:STDOUT:   %struct_type.field.8e0: type = struct_type {.field: %.308} [template]
// CHECK:STDOUT:   %struct: %struct_type.field.8e0 = struct_value (%.151) [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.5d6: %.5372ba.1 = splice_inst @A.MakeA.%tuple.elem0 [template]
// CHECK:STDOUT:   %A.c85d03.1: type = class_type @A, @A(%B) [concrete]
// CHECK:STDOUT:   %A.MakeA.type.509: type = fn_type @A.MakeA, @A(%B) [concrete]
// CHECK:STDOUT:   %A.MakeA.698: %A.MakeA.type.509 = struct_value () [concrete]
// CHECK:STDOUT:   %A.elem.1e6: type = unbound_element_type %A.c85d03.1, %B [concrete]
// CHECK:STDOUT:   %struct_type.field.8fb: type = struct_type {.field: %B} [concrete]
// CHECK:STDOUT:   %complete_type.08c: <witness> = complete_type_witness %struct_type.field.8fb [concrete]
// CHECK:STDOUT:   %inst.class_type: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %A.c85d03.2: type = class_type @A, @A(%B) [concrete = %A.c85d03.1]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.849: type = splice_block %A.c85d03.1 [concrete = %A.c85d03.1] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.386: Core.Form = init_form %A.c85d03.1 [concrete]
// CHECK:STDOUT:   %pattern_type.360: type = pattern_type %A.c85d03.1 [concrete]
// CHECK:STDOUT:   %return.param_patt.652: %pattern_type.360 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.b6c: %pattern_type.360 = return_slot_pattern %return.param_patt.652, %A.c85d03.1 [concrete]
// CHECK:STDOUT:   %A.MakeA.specific_fn: <specific function> = specific_function %A.MakeA.698, @A.MakeA(%B) [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_14.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc22_14.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.039: %pattern_type.e39 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.535: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt.039 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_14.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc22_14.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_14.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.79c: type = pattern_type %struct_type.field.8fb [concrete]
// CHECK:STDOUT:   %self.param_patt.df4: %pattern_type.79c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.20d: %pattern_type.79c = wrapper_binding_pattern self, %self.param_patt.df4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_14.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22_14.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.76c: %pattern_type.360 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.b24: %pattern_type.360 = wrapper_binding_pattern self, %self.param_patt.76c [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_14.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc22_14.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_14.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %MakeB.ref: %B.MakeB.type = name_ref MakeB, @B.%B.MakeB.decl [concrete = %B.MakeB]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fa4: init %B = splice_block %B.MakeB.call {
// CHECK:STDOUT:       %.272: ref %B = splice_block %.f2b {
// CHECK:STDOUT:         %.b39: ref %A.c85d03.1 = specific_inst @A.MakeA.%return.param, @A.MakeA(%B)
// CHECK:STDOUT:         %.f2b: ref %B = class_element_access %.b39, element0
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %B.MakeB.call: init %B to %.272 = call %B.MakeB()
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.215: init %A.c85d03.1 = splice_block %.831 {
// CHECK:STDOUT:       %.862: %struct_type.field.8fb = specific_inst @A.MakeA.%.loc13_31, @A.MakeA(%B)
// CHECK:STDOUT:       %.097: init %B = specific_inst @A.MakeA.%.loc13_30.2, @A.MakeA(%B)
// CHECK:STDOUT:       %.831: init %A.c85d03.1 to %.b39 = class_init (%.097)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_14.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc22_14.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_14.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.535]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %B = ref_param call_param0
// CHECK:STDOUT:     %self: ref %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc22_14.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.535]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %B = ref_param call_param0
// CHECK:STDOUT:     %self: ref %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_14.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.79c = ref_param_pattern [concrete = constants.%self.param_patt.df4]
// CHECK:STDOUT:     %self.patt: %pattern_type.79c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.field.8fb = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.field.8fb = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22_14.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.79c = ref_param_pattern [concrete = constants.%self.param_patt.df4]
// CHECK:STDOUT:     %self.patt: %pattern_type.79c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.field.8fb = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.field.8fb = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_14.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.360 = ref_param_pattern [concrete = constants.%self.param_patt.76c]
// CHECK:STDOUT:     %self.patt: %pattern_type.360 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b24]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %A.c85d03.1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %A.c85d03.1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc22_14.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.360 = ref_param_pattern [concrete = constants.%self.param_patt.76c]
// CHECK:STDOUT:     %self.patt: %pattern_type.360 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.b24]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %A.c85d03.1 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %A.c85d03.1 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc10_19.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.MakeA.type: type = fn_type @A.MakeA, @A(%T.loc10_19.1) [template = %A.MakeA.type (constants.%A.MakeA.type.27a)]
// CHECK:STDOUT:   %A.MakeA: @A.%A.MakeA.type (%A.MakeA.type.27a) = struct_value () [template = %A.MakeA (constants.%A.MakeA.f73)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %A.MakeA.decl: @A.%A.MakeA.type (%A.MakeA.type.27a) = fn_decl @A.MakeA [template = @A.%A.MakeA (constants.%A.MakeA.f73)] {
// CHECK:STDOUT:       %return.param_patt.loc12_20.1: @A.MakeA.%pattern_type (%pattern_type.fcb8f3.1) = out_param_pattern [template = %return.param_patt.loc12_20.2 (constants.%return.param_patt.2e9ac5.1)]
// CHECK:STDOUT:       %return.patt.loc12_14.1: @A.MakeA.%pattern_type (%pattern_type.fcb8f3.1) = return_slot_pattern %return.param_patt.loc12_20.1, %.loc12_20.9 [template = %return.patt.loc12_14.2 (constants.%return.patt.7fc6a1.1)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:       %T.ref.loc12: type = name_ref T, @A.%T.loc10_19.2 [template = %T (constants.%T)]
// CHECK:STDOUT:       %.loc12_20.7: type = type_of_inst %.loc12_20.1 [template = %.loc12_20.2 (constants.%.972)]
// CHECK:STDOUT:       %.loc12_20.8: @A.MakeA.%.loc12_20.2 (%.972) = splice_inst %.loc12_20.1 [template = %.loc12_20.3 (constants.%.a5e)]
// CHECK:STDOUT:       %.loc12_20.9: type = splice_inst %.loc12_20.4 [template = %.loc12_20.5 (constants.%.5372ba.1)]
// CHECK:STDOUT:       %.loc12_20.10: Core.Form = init_form %.loc12_20.9 [template = %.loc12_20.6 (constants.%.cd79f1.1)]
// CHECK:STDOUT:       %return.param: ref @A.MakeA.%.loc12_20.5 (%.5372ba.1) = out_param call_param0
// CHECK:STDOUT:       %return: ref @A.MakeA.%.loc12_20.5 (%.5372ba.1) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc18_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A.98c
// CHECK:STDOUT:     .A = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .MakeA = %A.MakeA.decl
// CHECK:STDOUT:     .field = %field_decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @A.MakeA(@A.%T.loc10_19.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template = %T (constants.%T)]
// CHECK:STDOUT:   %.loc12_20.1: <instruction> = call_action (%A.ref, %T.ref.loc12), false [template]
// CHECK:STDOUT:   %.loc12_20.2: type = type_of_inst %.loc12_20.1 [template = %.loc12_20.2 (constants.%.972)]
// CHECK:STDOUT:   %.loc12_20.3: @A.MakeA.%.loc12_20.2 (%.972) = splice_inst %.loc12_20.1 [template = %.loc12_20.3 (constants.%.a5e)]
// CHECK:STDOUT:   %.loc12_20.4: <instruction> = convert_to_value_action %.loc12_20.8, type [template]
// CHECK:STDOUT:   %.loc12_20.5: type = splice_inst %.loc12_20.4 [template = %.loc12_20.5 (constants.%.5372ba.1)]
// CHECK:STDOUT:   %.loc12_20.6: Core.Form = init_form %.loc12_20.5 [template = %.loc12_20.6 (constants.%.cd79f1.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.loc12_20.5 [template = %pattern_type (constants.%pattern_type.fcb8f3.1)]
// CHECK:STDOUT:   %return.param_patt.loc12_20.2: @A.MakeA.%pattern_type (%pattern_type.fcb8f3.1) = out_param_pattern [template = %return.param_patt.loc12_20.2 (constants.%return.param_patt.2e9ac5.1)]
// CHECK:STDOUT:   %return.patt.loc12_14.2: @A.MakeA.%pattern_type (%pattern_type.fcb8f3.1) = return_slot_pattern %return.param_patt.loc12_20.2, %.loc12_20.5 [template = %return.patt.loc12_14.2 (constants.%return.patt.7fc6a1.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %.loc12_20.5 [template = %require_complete (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %.loc13_23.3: <instruction> = access_member_action %T.ref.loc13, MakeB [template]
// CHECK:STDOUT:   %.loc13_23.4: type = type_of_inst %.loc13_23.3 [template = %.loc13_23.4 (constants.%.a08)]
// CHECK:STDOUT:   %.loc13_23.5: @A.MakeA.%.loc13_23.4 (%.a08) = splice_inst %.loc13_23.3 [template = %.loc13_23.5 (constants.%.5c2)]
// CHECK:STDOUT:   %.loc13_30.3: <instruction> = call_action (%.loc13_23.2), false [template]
// CHECK:STDOUT:   %.loc13_30.4: type = type_of_inst %.loc13_30.3 [template = %.loc13_30.4 (constants.%.308)]
// CHECK:STDOUT:   %.loc13_30.5: @A.MakeA.%.loc13_30.4 (%.308) = splice_inst %.loc13_30.3 [template = %.loc13_30.5 (constants.%.151)]
// CHECK:STDOUT:   %struct_type.field: type = struct_type {.field: @A.MakeA.%.loc13_30.4 (%.308)} [template = %struct_type.field (constants.%struct_type.field.8e0)]
// CHECK:STDOUT:   %struct: @A.MakeA.%struct_type.field (%struct_type.field.8e0) = struct_value (%.loc13_30.5) [template = %struct (constants.%struct)]
// CHECK:STDOUT:   %.loc13_32.2: %tuple.type = initialize_action %.loc13_31, %.loc12_20.5, %return.param, none, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc13_32.2, element0 [template]
// CHECK:STDOUT:   %.loc13_32.3: @A.MakeA.%.loc12_20.5 (%.5372ba.1) = splice_inst %tuple.elem0 [template = %.loc13_32.3 (constants.%.5d6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @A.MakeA.%.loc12_20.5 (%.5372ba.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, @A.%T.loc10_19.2 [template = %T (constants.%T)]
// CHECK:STDOUT:     %.loc13_23.1: type = type_of_inst %.loc13_23.3 [template = %.loc13_23.4 (constants.%.a08)]
// CHECK:STDOUT:     %.loc13_23.2: @A.MakeA.%.loc13_23.4 (%.a08) = splice_inst %.loc13_23.3 [template = %.loc13_23.5 (constants.%.5c2)]
// CHECK:STDOUT:     %.loc13_30.1: type = type_of_inst %.loc13_30.3 [template = %.loc13_30.4 (constants.%.308)]
// CHECK:STDOUT:     %.loc13_30.2: @A.MakeA.%.loc13_30.4 (%.308) = splice_inst %.loc13_30.3 [template = %.loc13_30.5 (constants.%.151)]
// CHECK:STDOUT:     %.loc13_31: @A.MakeA.%struct_type.field (%struct_type.field.8e0) = struct_literal (%.loc13_30.2) [template = %struct (constants.%struct)]
// CHECK:STDOUT:     %.loc13_32.1: @A.MakeA.%.loc12_20.5 (%.5372ba.1) = splice_inst %tuple.elem0 [template = %.loc13_32.3 (constants.%.5d6)]
// CHECK:STDOUT:     return %.loc13_32.1 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %A.ref: %A.type = name_ref A, file.%A.decl [concrete = constants.%A.generic]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %A: type = class_type @A, @A(constants.%B) [concrete = constants.%A.c85d03.1]
// CHECK:STDOUT:   %.loc22_7: %A.MakeA.type.509 = specific_constant @A.%A.MakeA.decl, @A(constants.%B) [concrete = constants.%A.MakeA.698]
// CHECK:STDOUT:   %MakeA.ref: %A.MakeA.type.509 = name_ref MakeA, %.loc22_7 [concrete = constants.%A.MakeA.698]
// CHECK:STDOUT:   %A.MakeA.specific_fn: <specific function> = specific_function %MakeA.ref, @A.MakeA(constants.%B) [concrete = constants.%A.MakeA.specific_fn]
// CHECK:STDOUT:   %.loc22_14.1: ref %A.c85d03.1 = temporary_storage
// CHECK:STDOUT:   %A.MakeA.call: init %A.c85d03.1 to %.loc22_14.1 = call %A.MakeA.specific_fn()
// CHECK:STDOUT:   %.loc22_14.2: ref %A.c85d03.1 = temporary %.loc22_14.1, %A.MakeA.call
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22_14.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22_14.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_14.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_14.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_14.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_14.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_14.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_14.2(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_14.3(%self.param: ref %struct_type.field.8fb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22: ref %B = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_14.3(%self.param: ref %struct_type.field.8fb) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_14.3(%self.param: ref %struct_type.field.8fb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_14.4(%self.param: ref %A.c85d03.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %field.ref: %A.elem.1e6 = name_ref field, @A.%field_decl [concrete = @A.%field_decl]
// CHECK:STDOUT:   %.loc22: ref %B = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_14.4(%self.param: ref %A.c85d03.1) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_14.4(%self.param: ref %A.c85d03.1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc10_19.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_19.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.MakeA.type => constants.%A.MakeA.type.27a
// CHECK:STDOUT:   %A.MakeA => constants.%A.MakeA.f73
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %A => constants.%A.98c
// CHECK:STDOUT:   %A.elem => constants.%A.elem.754
// CHECK:STDOUT:   %struct_type.field => constants.%struct_type.field.686
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.725
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.MakeA(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %.loc12_20.1 => constants.%.293
// CHECK:STDOUT:   %.loc12_20.2 => constants.%.3aa
// CHECK:STDOUT:   %.loc12_20.3 => constants.%.f2d
// CHECK:STDOUT:   %.loc12_20.4 => constants.%.c8c
// CHECK:STDOUT:   %.loc12_20.5 => constants.%.5372ba.2
// CHECK:STDOUT:   %.loc12_20.6 => constants.%.cd79f1.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fcb8f3.2
// CHECK:STDOUT:   %return.param_patt.loc12_20.2 => constants.%return.param_patt.2e9ac5.2
// CHECK:STDOUT:   %return.patt.loc12_14.2 => constants.%return.patt.7fc6a1.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%B) {
// CHECK:STDOUT:   %T.patt.loc10_19.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc10_19.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %A.MakeA.type => constants.%A.MakeA.type.509
// CHECK:STDOUT:   %A.MakeA => constants.%A.MakeA.698
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %A => constants.%A.c85d03.1
// CHECK:STDOUT:   %A.elem => constants.%A.elem.1e6
// CHECK:STDOUT:   %struct_type.field => constants.%struct_type.field.8fb
// CHECK:STDOUT:   %complete_type.loc18_1.2 => constants.%complete_type.08c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A.MakeA(constants.%B) {
// CHECK:STDOUT:   %T => constants.%B
// CHECK:STDOUT:   %.loc12_20.1 => constants.%inst.class_type
// CHECK:STDOUT:   %.loc12_20.2 => type
// CHECK:STDOUT:   %.loc12_20.3 => constants.%A.c85d03.1
// CHECK:STDOUT:   %.loc12_20.4 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc12_20.5 => constants.%A.c85d03.1
// CHECK:STDOUT:   %.loc12_20.6 => constants.%.386
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.360
// CHECK:STDOUT:   %return.param_patt.loc12_20.2 => constants.%return.param_patt.652
// CHECK:STDOUT:   %return.patt.loc12_14.2 => constants.%return.patt.b6c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.08c
// CHECK:STDOUT:   %.loc13_23.3 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc13_23.4 => constants.%B.MakeB.type
// CHECK:STDOUT:   %.loc13_23.5 => constants.%B.MakeB
// CHECK:STDOUT:   %.loc13_30.3 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc13_30.4 => constants.%B
// CHECK:STDOUT:   %.loc13_30.5 => <not constant>
// CHECK:STDOUT:   %struct_type.field => constants.%struct_type.field.8fb
// CHECK:STDOUT:   %struct => <not constant>
// CHECK:STDOUT:   %.loc13_32.2 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc13_32.3 => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- initialize_from_template_method_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %pattern_type.e39: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %B.MakeB.type: type = fn_type @B.MakeB [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B.MakeB: %B.MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %A.elem: type = unbound_element_type %A, %B [concrete]
// CHECK:STDOUT:   %struct_type.field.8fb: type = struct_type {.field: %B} [concrete]
// CHECK:STDOUT:   %complete_type.08c: <witness> = complete_type_witness %struct_type.field.8fb [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0, template [template]
// CHECK:STDOUT:   %.cbf: Core.Form = init_form %A [concrete]
// CHECK:STDOUT:   %pattern_type.9ef: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %return.param_patt.f0c: %pattern_type.9ef = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a6d: %pattern_type.9ef = return_slot_pattern %return.param_patt.f0c, %A [concrete]
// CHECK:STDOUT:   %MakeA.type: type = fn_type @MakeA [concrete]
// CHECK:STDOUT:   %MakeA: %MakeA.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a08: type = type_of_inst @MakeA.%.loc17_16.3 [template]
// CHECK:STDOUT:   %.5c2: %.a08 = splice_inst @MakeA.%.loc17_16.3 [template]
// CHECK:STDOUT:   %.308: type = type_of_inst @MakeA.%.loc17_23.3 [template]
// CHECK:STDOUT:   %.151: %.308 = splice_inst @MakeA.%.loc17_23.3 [template]
// CHECK:STDOUT:   %struct_type.field.8e0: type = struct_type {.field: %.308} [template]
// CHECK:STDOUT:   %struct: %struct_type.field.8e0 = struct_value (%.151) [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a21: ref <instruction> = splice_block %.f4aa6f.2 {
// CHECK:STDOUT:       %.f4aa6f.2: ref %B = class_element_access @MakeA.%return.param, element0
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.7e2: %B = splice_inst @MakeA.%tuple.elem0 [template]
// CHECK:STDOUT:   %A.val: %A = struct_value (%.7e2) [template]
// CHECK:STDOUT:   %Run.type: type = fn_type @Run [concrete]
// CHECK:STDOUT:   %Run: %Run.type = struct_value () [concrete]
// CHECK:STDOUT:   %MakeA.specific_fn: <specific function> = specific_function %MakeA, @MakeA(%B) [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_10.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc22_10.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.039: %pattern_type.e39 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.535: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt.039 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_10.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc22_10.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_10.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.79c: type = pattern_type %struct_type.field.8fb [concrete]
// CHECK:STDOUT:   %self.param_patt.df4: %pattern_type.79c = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.20d: %pattern_type.79c = wrapper_binding_pattern self, %self.param_patt.df4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_10.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.3: type = fn_type @Destroy.WithSelf.Op.loc22_10.3 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.3: %Destroy.WithSelf.Op.type.ef016f.3 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.8fb: %pattern_type.9ef = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.e64: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt.8fb [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc22_10.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.4: type = fn_type @Destroy.WithSelf.Op.loc22_10.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.4: %Destroy.WithSelf.Op.type.ef016f.4 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.4: type = fn_type @Destroy.WithSelf.SelfDestruct.loc22_10.4 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.4: %Destroy.WithSelf.SelfDestruct.type.fbceb5.4 = struct_value () [concrete]
// CHECK:STDOUT:   %inst.name_ref: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %MakeB.ref: %B.MakeB.type = name_ref MakeB, @B.%B.MakeB.decl [concrete = %B.MakeB]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.772: init %B = splice_block %B.MakeB.call {
// CHECK:STDOUT:       %.f4aa6f.1: ref %B = class_element_access @MakeA.%return.param, element0
// CHECK:STDOUT:       %B.MakeB.call: init %B to %.f4aa6f.1 = call %B.MakeB()
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.3f4: init %B = splice_block %.772 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.3) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_10.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc22_10.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_10.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.535]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %B = ref_param call_param0
// CHECK:STDOUT:     %self: ref %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc22_10.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e39 = ref_param_pattern [concrete = constants.%self.param_patt.039]
// CHECK:STDOUT:     %self.patt: %pattern_type.e39 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.535]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %B = ref_param call_param0
// CHECK:STDOUT:     %self: ref %B = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.3: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.3 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_10.3 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.79c = ref_param_pattern [concrete = constants.%self.param_patt.df4]
// CHECK:STDOUT:     %self.patt: %pattern_type.79c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.field.8fb = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.field.8fb = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.3: %Destroy.WithSelf.Op.type.ef016f.3 = fn_decl @Destroy.WithSelf.Op.loc22_10.3 [concrete = constants.%Destroy.WithSelf.Op.403171.3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.79c = ref_param_pattern [concrete = constants.%self.param_patt.df4]
// CHECK:STDOUT:     %self.patt: %pattern_type.79c = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.20d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %struct_type.field.8fb = ref_param call_param0
// CHECK:STDOUT:     %self: ref %struct_type.field.8fb = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.4: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.4 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc22_10.4 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e64]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.4: %Destroy.WithSelf.Op.type.ef016f.4 = fn_decl @Destroy.WithSelf.Op.loc22_10.4 [concrete = constants.%Destroy.WithSelf.Op.403171.4] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9ef = ref_param_pattern [concrete = constants.%self.param_patt.8fb]
// CHECK:STDOUT:     %self.patt: %pattern_type.9ef = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.e64]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %A = ref_param call_param0
// CHECK:STDOUT:     %self: ref %A = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: type = class_decl @A [concrete = constants.%A] {} {}
// CHECK:STDOUT:   %MakeA.decl: %MakeA.type = fn_decl @MakeA [concrete = constants.%MakeA] {
// CHECK:STDOUT:     %T.patt.loc15_20.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc15_20.2 (constants.%T.patt)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9ef = out_param_pattern [concrete = constants.%return.param_patt.f0c]
// CHECK:STDOUT:     %return.patt: %pattern_type.9ef = return_slot_pattern %return.param_patt, %A.ref [concrete = constants.%return.patt.a6d]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %A.ref: type = name_ref A, file.%A.decl [concrete = constants.%A]
// CHECK:STDOUT:     %.loc15_31: Core.Form = init_form %A.ref [concrete = constants.%.cbf]
// CHECK:STDOUT:     %.loc15_22.1: type = splice_block %.loc15_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc15_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_20.2: type = symbolic_binding T, 0, template [template = %T.loc15_20.1 (constants.%T)]
// CHECK:STDOUT:     %return.param: ref %A = out_param call_param0
// CHECK:STDOUT:     %return: ref %A = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Run.decl: %Run.type = fn_decl @Run [concrete = constants.%Run] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A {
// CHECK:STDOUT:   %field_decl: %A.elem = field_decl field, element0, %B.ref in [concrete] {
// CHECK:STDOUT:     %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.field.8fb [concrete = constants.%complete_type.08c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A
// CHECK:STDOUT:   .B = <poisoned>
// CHECK:STDOUT:   .field = %field_decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @MakeA(%T.loc15_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc15_20.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0, template [template = %T.patt.loc15_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc15_20.1: type = symbolic_binding T, 0, template [template = %T.loc15_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc17_16.3: <instruction> = access_member_action %T.ref, MakeB [template]
// CHECK:STDOUT:   %.loc17_16.4: type = type_of_inst %.loc17_16.3 [template = %.loc17_16.4 (constants.%.a08)]
// CHECK:STDOUT:   %.loc17_16.5: @MakeA.%.loc17_16.4 (%.a08) = splice_inst %.loc17_16.3 [template = %.loc17_16.5 (constants.%.5c2)]
// CHECK:STDOUT:   %.loc17_23.3: <instruction> = call_action (%.loc17_16.2), false [template]
// CHECK:STDOUT:   %.loc17_23.4: type = type_of_inst %.loc17_23.3 [template = %.loc17_23.4 (constants.%.308)]
// CHECK:STDOUT:   %.loc17_23.5: @MakeA.%.loc17_23.4 (%.308) = splice_inst %.loc17_23.3 [template = %.loc17_23.5 (constants.%.151)]
// CHECK:STDOUT:   %struct_type.field: type = struct_type {.field: @MakeA.%.loc17_23.4 (%.308)} [template = %struct_type.field (constants.%struct_type.field.8e0)]
// CHECK:STDOUT:   %struct: @MakeA.%struct_type.field (%struct_type.field.8e0) = struct_value (%.loc17_23.5) [template = %struct (constants.%struct)]
// CHECK:STDOUT:   %.loc18_4.4: %tuple.type = initialize_action %.loc17_23.2, constants.%B, constants.%.f4aa6f.2, constants.%inst.splice_block.426619.1, true [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc18_4.4, element0 [template]
// CHECK:STDOUT:   %.loc18_4.5: %B = splice_inst %tuple.elem0 [template = %.loc18_4.5 (constants.%.7e2)]
// CHECK:STDOUT:   %A.val: %A = struct_value (%.loc18_4.5) [template = %A.val (constants.%A.val)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: %A {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_20.2 [template = %T.loc15_20.1 (constants.%T)]
// CHECK:STDOUT:     %.loc17_16.1: type = type_of_inst %.loc17_16.3 [template = %.loc17_16.4 (constants.%.a08)]
// CHECK:STDOUT:     %.loc17_16.2: @MakeA.%.loc17_16.4 (%.a08) = splice_inst %.loc17_16.3 [template = %.loc17_16.5 (constants.%.5c2)]
// CHECK:STDOUT:     %.loc17_23.1: type = type_of_inst %.loc17_23.3 [template = %.loc17_23.4 (constants.%.308)]
// CHECK:STDOUT:     %.loc17_23.2: @MakeA.%.loc17_23.4 (%.308) = splice_inst %.loc17_23.3 [template = %.loc17_23.5 (constants.%.151)]
// CHECK:STDOUT:     %.loc18_4.1: @MakeA.%struct_type.field (%struct_type.field.8e0) = struct_literal (%.loc17_23.2) [template = %struct (constants.%struct)]
// CHECK:STDOUT:     %.loc18_4.2: %B = splice_inst %tuple.elem0 [template = %.loc18_4.5 (constants.%.7e2)]
// CHECK:STDOUT:     %.loc18_4.3: init %A to %return.param = class_init (%.loc18_4.2) [template = %A.val (constants.%A.val)]
// CHECK:STDOUT:     %.loc18_5: init %A = converted %.loc18_4.1, %.loc18_4.3 [template = %A.val (constants.%A.val)]
// CHECK:STDOUT:     return %.loc18_5 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeA.ref: %MakeA.type = name_ref MakeA, file.%MakeA.decl [concrete = constants.%MakeA]
// CHECK:STDOUT:   %B.ref: type = name_ref B, file.%B.decl [concrete = constants.%B]
// CHECK:STDOUT:   %MakeA.specific_fn: <specific function> = specific_function %MakeA.ref, @MakeA(constants.%B) [concrete = constants.%MakeA.specific_fn]
// CHECK:STDOUT:   %.loc22_10.1: ref %A = temporary_storage
// CHECK:STDOUT:   %MakeA.call: init %A to %.loc22_10.1 = call %MakeA.specific_fn()
// CHECK:STDOUT:   %.loc22_10.2: ref %A = temporary %.loc22_10.1, %MakeA.call
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22_10.2, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.4
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22_10.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_10.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_10.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_10.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_10.2(%self.param: ref %B) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_10.2(%self.param: ref %B) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_10.3(%self.param: ref %struct_type.field.8fb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc22: ref %B = struct_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_10.3(%self.param: ref %struct_type.field.8fb) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_10.3(%self.param: ref %struct_type.field.8fb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.3(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.3(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc22_10.4(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %field.ref: %A.elem = name_ref field, @A.%field_decl [concrete = @A.%field_decl]
// CHECK:STDOUT:   %.loc22: ref %B = class_element_access %self.param, element0
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.loc22, constants.%Destroy.WithSelf.SelfDestruct.db3fdb.2
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %Destroy.WithSelf.SelfDestruct.bound(%.loc22)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc22_10.4(%self.param: ref %A) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc22_10.4(%self.param: ref %A) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.4(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.4(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeA(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc15_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @MakeA(constants.%B) {
// CHECK:STDOUT:   %T.patt.loc15_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc15_20.1 => constants.%B
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc17_16.3 => constants.%inst.name_ref
// CHECK:STDOUT:   %.loc17_16.4 => constants.%B.MakeB.type
// CHECK:STDOUT:   %.loc17_16.5 => constants.%B.MakeB
// CHECK:STDOUT:   %.loc17_23.3 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc17_23.4 => constants.%B
// CHECK:STDOUT:   %.loc17_23.5 => <not constant>
// CHECK:STDOUT:   %struct_type.field => constants.%struct_type.field.8fb
// CHECK:STDOUT:   %struct => <not constant>
// CHECK:STDOUT:   %.loc18_4.4 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc18_4.5 => <not constant>
// CHECK:STDOUT:   %A.val => <not constant>
// CHECK:STDOUT: }
// CHECK:STDOUT:

// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/impl_thunk_min_prelude.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/impl_thunk_min_prelude.carbon

// --- convert_in_class.carbon

library "[[@TEST_NAME]]";

interface X(T:! type, U:! type) {
  fn F(t: T) -> U;
}

class ConvertsToA {}
class ConvertsToB {}

// Check that we don't try to define a thunk for `A.B.(as X).F` until we reach
// the end of `A`. If we tried earlier, we wouldn't find a conversion from
// `ConvertsToA` to `A` or from `ConvertsToB` to `B`.
class A {
  class B {
    impl as X(ConvertsToA, B) {
      fn F(a: A) -> ConvertsToB;
    }

    impl ConvertsToB as Core.ImplicitAs(B) {
      fn Convert[unused self: Self]() -> B { return {}; }
    }
  }

  impl ConvertsToA as Core.ImplicitAs(A) {
    fn Convert[unused self: Self]() -> A { return {}; }
  }
}

// --- fail_todo_out_of_line_thunk.carbon

library "[[@TEST_NAME]]";

class Wrap(T:! type) {}

interface OpWith(U:! type) {
  fn Op[self: Self](u: U);
}

impl forall [T:! type, U:! Core.ImplicitAs(Wrap(T))] Wrap(T) as OpWith(U) {
  // CHECK:STDERR: fail_todo_out_of_line_thunk.carbon:[[@LINE+7]]:3: error: use of undefined generic function [MissingGenericFunctionDefinition]
  // CHECK:STDERR:   fn Op[self: Self](other: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_out_of_line_thunk.carbon:[[@LINE+4]]:3: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:   fn Op[self: Self](other: Self);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op[self: Self](other: Self);
}

// TODO: Once we support the syntax for defining impl members out of line,
// define the above function here.
// fn (forall [T:! type, U:! Core.ImplicitAs(Wrap(T))] Wrap(T) as OpWith(U)).Op[self: Self](other: Self) {}

// --- builtin_thunk.carbon

library "[[@TEST_NAME]]";

class Wrap(T:! type) {}

interface OpWith(U:! type) {
  fn Op[self: Self](u: U);
}

impl forall [T:! type, U:! Core.ImplicitAs(Wrap(T))] Wrap(T) as OpWith(U) {
  fn Op[self: Self](other: Self) = "no_op";
}

// --- thunk_literal_convert.carbon

library "[[@TEST_NAME]]";

interface Add(T:! type) {
  fn Op(a: Self, b: T) -> Self;
}
impl forall [T:! Core.ImplicitAs(i32)] i32 as Add(T) {
  fn Op(a: Self, b: Self) -> Self = "int.sadd";
}

fn Call() -> i32 {
  let a: i32 = 1;
  // The conversion from 2 to IntLiteral here relies on having the
  // constant value available, so is only possible if the thunk is
  // inlined.
  return i32.(Add(Core.IntLiteral).Op)
  //@dump-sem-ir-begin
        (a, 2)
  //@dump-sem-ir-end
  ;
}

// CHECK:STDOUT: --- thunk_literal_convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %i32.as.Add.impl.Op.type.79f953.1: type = fn_type @i32.as.Add.impl.Op.loc8_35.1, @i32.as.Add.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %i32.as.Add.impl.Op.0c3802.1: %i32.as.Add.impl.Op.type.79f953.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %i32.as.Add.impl.Op.specific_fn.f5e09b.2: <specific function> = specific_function %i32.as.Add.impl.Op.0c3802.1, @i32.as.Add.impl.Op.loc8_35.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.3cb: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.295: %i32 = int_value 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: %i32 = name_ref a, %a
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness.a23) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc18_14.1: %ImplicitAs.type.914 = converted Core.IntLiteral, %ImplicitAs.facet [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc18_14.2: %i32.as.Add.impl.Op.type.79f953.1 = specific_constant @i32.as.Add.impl.%i32.as.Add.impl.Op.decl.loc8_35.1, @i32.as.Add.impl(constants.%ImplicitAs.facet) [concrete = constants.%i32.as.Add.impl.Op.0c3802.1]
// CHECK:STDOUT:   %Op.ref.loc18: %i32.as.Add.impl.Op.type.79f953.1 = name_ref Op, %.loc18_14.2 [concrete = constants.%i32.as.Add.impl.Op.0c3802.1]
// CHECK:STDOUT:   %i32.as.Add.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc18, @i32.as.Add.impl.Op.loc8_35.1(constants.%ImplicitAs.facet) [concrete = constants.%i32.as.Add.impl.Op.specific_fn.f5e09b.2]
// CHECK:STDOUT:   %impl.elem0.loc18: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc18_13.1: <bound method> = bound_method %int_2, %impl.elem0.loc18 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.9f3]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_13.2: <bound method> = bound_method %int_2, %specific_fn.loc18 [concrete = constants.%bound_method.3cb]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18: init %i32 = call %bound_method.loc18_13.2(%int_2) [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc18_13.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc18 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %.loc18_13.2: %i32 = converted %int_2, %.loc18_13.1 [concrete = constants.%int_2.295]
// CHECK:STDOUT:   %i32.as.Add.impl.Op.call: init %i32 = call %i32.as.Add.impl.Op.specific_fn(%a.ref, %.loc18_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

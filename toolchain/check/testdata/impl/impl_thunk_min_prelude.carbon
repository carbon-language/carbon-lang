// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
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
      fn Convert[self: Self]() -> B { return {}; }
    }
  }

  impl ConvertsToA as Core.ImplicitAs(A) {
    fn Convert[self: Self]() -> A { return {}; }
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

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(size: IntLiteral()) -> type = "int.make_type_signed";

impl IntLiteral() as Core.ImplicitAs(Int(32)) {
  fn Convert[self: Self]() -> Int(32) = "int.convert_checked";
}

interface Add(T:! type) {
  fn Op(a: Self, b: T) -> Self;
}
impl forall [T:! Core.ImplicitAs(Int(32))] Int(32) as Add(T) {
  fn Op(a: Self, b: Self) -> Self = "int.sadd";
}

fn Call() -> Int(32) {
  let a: Int(32) = 1;
  // The conversion from 2 to IntLiteral here relies on having the
  // constant value available, so is only possible if the thunk is
  // inlined.
  return Int(32).(Add(IntLiteral()).Op)
  //@dump-sem-ir-begin
        (a, 2)
  //@dump-sem-ir-end
  ;
}

// CHECK:STDOUT: --- thunk_literal_convert.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.873: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.059: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness file.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.873 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %.0f5: type = fn_type_with_self_type %ImplicitAs.Convert.type.059, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %i32.builtin.as.Add.impl.Op.type.5284bd.1: type = fn_type @i32.builtin.as.Add.impl.Op.loc15_35.1, @i32.builtin.as.Add.impl(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %i32.builtin.as.Add.impl.Op.1d2eda.1: %i32.builtin.as.Add.impl.Op.type.5284bd.1 = struct_value () [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e54: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.Add.impl.Op.specific_fn.ffd5e9.2: <specific function> = specific_function %i32.builtin.as.Add.impl.Op.1d2eda.1, @i32.builtin.as.Add.impl.Op.loc15_35.1(%ImplicitAs.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Call() -> %i32.builtin {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %a.ref: %i32.builtin = name_ref a, %a
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_14.1: %ImplicitAs.type.873 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc25_14.1: %ImplicitAs.type.873 = converted Core.IntLiteral, %ImplicitAs.facet.loc25_14.1 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %ImplicitAs.facet.loc25_14.2: %ImplicitAs.type.873 = facet_value Core.IntLiteral, (constants.%ImplicitAs.impl_witness) [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   %.loc25_14.2: %ImplicitAs.type.873 = converted Core.IntLiteral, %ImplicitAs.facet.loc25_14.2 [concrete = constants.%ImplicitAs.facet]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc25_14.3: %i32.builtin.as.Add.impl.Op.type.5284bd.1 = specific_constant @i32.builtin.as.Add.impl.%i32.builtin.as.Add.impl.Op.decl.loc15_35.1, @i32.builtin.as.Add.impl(constants.%ImplicitAs.facet) [concrete = constants.%i32.builtin.as.Add.impl.Op.1d2eda.1]
// CHECK:STDOUT:   %Op.ref.loc25: %i32.builtin.as.Add.impl.Op.type.5284bd.1 = name_ref Op, %.loc25_14.3 [concrete = constants.%i32.builtin.as.Add.impl.Op.1d2eda.1]
// CHECK:STDOUT:   %impl.elem0.loc25_14: %.0f5 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_14: <bound method> = bound_method %int_2, %impl.elem0.loc25_14 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e54]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_14: init %i32.builtin = call %bound_method.loc25_14(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc25_14.4: %i32.builtin = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_14 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc25_14.5: %i32.builtin = converted %int_2, %.loc25_14.4 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %i32.builtin.as.Add.impl.Op.specific_fn: <specific function> = specific_function %Op.ref.loc25, @i32.builtin.as.Add.impl.Op.loc15_35.1(constants.%ImplicitAs.facet) [concrete = constants.%i32.builtin.as.Add.impl.Op.specific_fn.ffd5e9.2]
// CHECK:STDOUT:   %impl.elem0.loc25_13: %.0f5 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc25_13: <bound method> = bound_method %int_2, %impl.elem0.loc25_13 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.e54]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_13: init %i32.builtin = call %bound_method.loc25_13(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc25_13.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc25_13 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %.loc25_13.2: %i32.builtin = converted %int_2, %.loc25_13.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:   %i32.builtin.as.Add.impl.Op.call: init %i32.builtin = call %i32.builtin.as.Add.impl.Op.specific_fn(%a.ref, %.loc25_13.2)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:

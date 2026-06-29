// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/builtin_candidates.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/builtin_candidates.carbon

// --- enum_operators.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
enum E { e1, e2, e3 };
enum class EC { e1, e2, e3 };
''';

fn TestEnum(a: Cpp.E, b: Cpp.E) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

fn TestEnumClass(a: Cpp.EC, b: Cpp.EC) {
  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

// --- fail_enum_class_bitwise_operators.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
enum class EC { e1, e2, e3 };
''';

fn TestEnumClass(a: Cpp.EC, b: Cpp.EC) {
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.BitAndWith(Cpp.EC)` in type `Cpp.EC` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: Cpp.EC = a & b;
  // CHECK:STDERR:                   ^~~~~
  // CHECK:STDERR:
  let _: Cpp.EC = a & b;
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.BitOrWith(Cpp.EC)` in type `Cpp.EC` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: Cpp.EC = a | b;
  // CHECK:STDERR:                   ^~~~~
  // CHECK:STDERR:
  let _: Cpp.EC = a | b;
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.BitXorWith(Cpp.EC)` in type `Cpp.EC` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: Cpp.EC = a ^ b;
  // CHECK:STDERR:                   ^~~~~
  // CHECK:STDERR:
  let _: Cpp.EC = a ^ b;
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:19: error: cannot access member of interface `Core.BitComplement` in type `Cpp.EC` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: Cpp.EC = ^a;
  // CHECK:STDERR:                   ^~
  // CHECK:STDERR:
  let _: Cpp.EC = ^a;
}

// --- enum_operators_with_conversion.carbon

import Cpp inline '''c++
enum E {};
struct ConvertibleToEnum {
  operator E() const;
};
''';

fn TestEnumVsConvertibleToEnum(a: Cpp.E, b: Cpp.ConvertibleToEnum) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

fn TestConvertibleToEnumVsEnum(a: Cpp.ConvertibleToEnum, b: Cpp.E) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

// TODO: Should we allow these? For consistency with the cases in the next
// split, we should not, but we currently do.
fn TestConversionOnBothOperands(a: Cpp.ConvertibleToEnum, b: Cpp.E) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

// --- fail_convert_all_operands.carbon

// We do not currently support calls to builtin candidates where all operands
// are converted from class type to a primitive type. This is done to match
// Carbon rules.

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct ConvertibleToInt {
  operator int() const;
};
struct ConvertibleToFloat {
  operator float() const;
};
''';

fn TestInteger(a: Cpp.ConvertibleToInt, b: Cpp.ConvertibleToInt) {
  // Test arithmetic.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.AddWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a + b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a + b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.SubWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a - b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a - b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.MulWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a * b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a * b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.DivWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a / b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a / b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.ModWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a % b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a % b;

  // Test bitwise.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.BitAndWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a & b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a & b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.BitOrWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a | b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a | b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.BitXorWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a ^ b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: i32 = a ^ b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.LeftShiftWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a << b;
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR:
  let _: i32 = a << b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.RightShiftWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = a >> b;
  // CHECK:STDERR:                ^~~~~~
  // CHECK:STDERR:
  let _: i32 = a >> b;

  // Test comparison.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.EqWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a == b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a == b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.EqWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a != b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a != b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a < b);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  let _: bool = (a < b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a <= b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a <= b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a > b);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  let _: bool = (a > b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToInt)` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a >= b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.Negate` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = -a;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  let _: i32 = -a;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.BitComplement` in type `Cpp.ConvertibleToInt` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: i32 = ^a;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  let _: i32 = ^a;
}

fn TestFloat(a: Cpp.ConvertibleToFloat, b: Cpp.ConvertibleToFloat) {
  // Test arithmetic.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.AddWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: f32 = a + b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: f32 = a + b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.SubWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: f32 = a - b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: f32 = a - b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.MulWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: f32 = a * b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: f32 = a * b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.DivWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: f32 = a / b;
  // CHECK:STDERR:                ^~~~~
  // CHECK:STDERR:
  let _: f32 = a / b;

  // Test comparison.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.EqWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a == b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a == b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.EqWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a != b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a != b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a < b);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  let _: bool = (a < b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a <= b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a <= b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a > b);
  // CHECK:STDERR:                  ^~~~~
  // CHECK:STDERR:
  let _: bool = (a > b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: error: cannot access member of interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: bool = (a >= b);
  // CHECK:STDERR:                  ^~~~~~
  // CHECK:STDERR:
  let _: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: cannot access member of interface `Core.Negate` in type `Cpp.ConvertibleToFloat` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   let _: f32 = -a;
  // CHECK:STDERR:                ^~
  // CHECK:STDERR:
  let _: f32 = -a;
}

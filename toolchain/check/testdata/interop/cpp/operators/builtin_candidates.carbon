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
  // CHECK:STDERR:        ╭─┤ fail_enum_class_bitwise_operators.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   let _: Cpp.EC = a & b;
  // CHECK:STDERR:        ·                   ┬ ┳ ┬
  // CHECK:STDERR:        ·                   │ ┗━━━━┫ error: type `Cpp.EC` does not implement interface
  // CHECK:STDERR:        ·                   │   │  ┃        `Core.BitAndWith(Cpp.EC)`
  // CHECK:STDERR:        ·                   │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                   ╰────────┤ left operand has type `EC` [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `EC`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.EC = a & b;
  // CHECK:STDERR:        ╭─┤ fail_enum_class_bitwise_operators.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   let _: Cpp.EC = a | b;
  // CHECK:STDERR:        ·                   ┬ ┳ ┬
  // CHECK:STDERR:        ·                   │ ┗━━━━┫ error: type `Cpp.EC` does not implement interface
  // CHECK:STDERR:        ·                   │   │  ┃        `Core.BitOrWith(Cpp.EC)`
  // CHECK:STDERR:        ·                   │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                   ╰────────┤ left operand has type `EC` [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `EC`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.EC = a | b;
  // CHECK:STDERR:        ╭─┤ fail_enum_class_bitwise_operators.carbon:[[@LINE+11]]:21
  // CHECK:STDERR: ->     │   let _: Cpp.EC = a ^ b;
  // CHECK:STDERR:        ·                   ┬ ┳ ┬
  // CHECK:STDERR:        ·                   │ ┗━━━━┫ error: type `Cpp.EC` does not implement interface
  // CHECK:STDERR:        ·                   │   │  ┃        `Core.BitXorWith(Cpp.EC)`
  // CHECK:STDERR:        ·                   │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                   ╰────────┤ left operand has type `EC` [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `EC`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: Cpp.EC = a ^ b;
  // CHECK:STDERR:        ╭─┤ fail_enum_class_bitwise_operators.carbon:[[@LINE+8]]:19
  // CHECK:STDERR: ->     │   let _: Cpp.EC = ^a;
  // CHECK:STDERR:        ·                   ┳
  // CHECK:STDERR:        ·                   ┗━━┫ error: type `Cpp.EC` does not implement interface
  // CHECK:STDERR:        ·                      ┃        `Core.BitComplement`
  // CHECK:STDERR:        ·                      ┃        [MissingImplForType]
  // CHECK:STDERR:        ╰────
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
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a + b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.AddWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a + b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a - b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.SubWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a - b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a * b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.MulWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a * b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a / b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.DivWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a / b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a % b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.ModWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a % b;

  // Test bitwise.
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a & b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.BitAndWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a & b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a | b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.BitOrWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a | b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a ^ b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.BitXorWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a ^ b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a << b;
  // CHECK:STDERR:        ·                ┬ ━┳ ┬
  // CHECK:STDERR:        ·                │  ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │    │  ┃        interface
  // CHECK:STDERR:        ·                │    │  ┃        `Core.LeftShiftWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰─────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                     │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                     ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                            │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a << b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:18
  // CHECK:STDERR: ->     │   let _: i32 = a >> b;
  // CHECK:STDERR:        ·                ┬ ━┳ ┬
  // CHECK:STDERR:        ·                │  ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                │    │  ┃        interface
  // CHECK:STDERR:        ·                │    │  ┃        `Core.RightShiftWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰─────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                     │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                     ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                            │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = a >> b;

  // Test comparison.
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a == b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                  │    │  ┃        interface `Core.EqWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a == b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a != b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                  │    │  ┃        interface `Core.EqWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a != b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a < b);
  // CHECK:STDERR:        ·                  ┬ ┳ ┬
  // CHECK:STDERR:        ·                  │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                  │   │  ┃        interface
  // CHECK:STDERR:        ·                  │   │  ┃        `Core.OrderedWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                  │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                      │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                      ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                             │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a < b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a <= b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                  │    │  ┃        interface
  // CHECK:STDERR:        ·                  │    │  ┃        `Core.OrderedWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a <= b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a > b);
  // CHECK:STDERR:        ·                  ┬ ┳ ┬
  // CHECK:STDERR:        ·                  │ ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                  │   │  ┃        interface
  // CHECK:STDERR:        ·                  │   │  ┃        `Core.OrderedWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                  │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                      │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                      ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                             │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a > b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a >= b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                  │    │  ┃        interface
  // CHECK:STDERR:        ·                  │    │  ┃        `Core.OrderedWith(Cpp.ConvertibleToInt)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToInt`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+8]]:16
  // CHECK:STDERR: ->     │   let _: i32 = -a;
  // CHECK:STDERR:        ·                ┳
  // CHECK:STDERR:        ·                ┗━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                   ┃        interface `Core.Negate`
  // CHECK:STDERR:        ·                   ┃        [MissingImplForType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = -a;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+8]]:16
  // CHECK:STDERR: ->     │   let _: i32 = ^a;
  // CHECK:STDERR:        ·                ┳
  // CHECK:STDERR:        ·                ┗━━┫ error: type `Cpp.ConvertibleToInt` does not implement
  // CHECK:STDERR:        ·                   ┃        interface `Core.BitComplement`
  // CHECK:STDERR:        ·                   ┃        [MissingImplForType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: i32 = ^a;
}

fn TestFloat(a: Cpp.ConvertibleToFloat, b: Cpp.ConvertibleToFloat) {
  // Test arithmetic.
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: f32 = a + b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.AddWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: f32 = a + b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: f32 = a - b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.SubWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: f32 = a - b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: f32 = a * b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.MulWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: f32 = a * b;
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+12]]:18
  // CHECK:STDERR: ->     │   let _: f32 = a / b;
  // CHECK:STDERR:        ·                ┬ ┳ ┬
  // CHECK:STDERR:        ·                │ ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not implement
  // CHECK:STDERR:        ·                │   │  ┃        interface `Core.DivWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                ╰────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                    │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                    ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                           │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: f32 = a / b;

  // Test comparison.
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a == b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not
  // CHECK:STDERR:        ·                  │    │  ┃        implement interface
  // CHECK:STDERR:        ·                  │    │  ┃        `Core.EqWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a == b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a != b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not
  // CHECK:STDERR:        ·                  │    │  ┃        implement interface
  // CHECK:STDERR:        ·                  │    │  ┃        `Core.EqWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a != b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a < b);
  // CHECK:STDERR:        ·                  ┬ ┳ ┬
  // CHECK:STDERR:        ·                  │ ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not
  // CHECK:STDERR:        ·                  │   │  ┃        implement interface
  // CHECK:STDERR:        ·                  │   │  ┃        `Core.OrderedWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                  │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                      │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                      ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                             │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a < b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a <= b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not
  // CHECK:STDERR:        ·                  │    │  ┃        implement interface
  // CHECK:STDERR:        ·                  │    │  ┃        `Core.OrderedWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a <= b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a > b);
  // CHECK:STDERR:        ·                  ┬ ┳ ┬
  // CHECK:STDERR:        ·                  │ ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not
  // CHECK:STDERR:        ·                  │   │  ┃        implement interface
  // CHECK:STDERR:        ·                  │   │  ┃        `Core.OrderedWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                  │   │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                      │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                      ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                             │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a > b);
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+13]]:20
  // CHECK:STDERR: ->     │   let _: bool = (a >= b);
  // CHECK:STDERR:        ·                  ┬ ━┳ ┬
  // CHECK:STDERR:        ·                  │  ┗━━━━┫ error: type `Cpp.ConvertibleToFloat` does not
  // CHECK:STDERR:        ·                  │    │  ┃        implement interface
  // CHECK:STDERR:        ·                  │    │  ┃        `Core.OrderedWith(Cpp.ConvertibleToFloat)`
  // CHECK:STDERR:        ·                  │    │  ┃        [MissingImplForType]
  // CHECK:STDERR:        ·                  ╰─────────┤ left operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                       │    │ [BinaryOperatorLhsType]
  // CHECK:STDERR:        ·                       ╰──────┤ right operand has type `ConvertibleToFloat`
  // CHECK:STDERR:        ·                              │ [BinaryOperatorRhsType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR:        ╭─┤ fail_convert_all_operands.carbon:[[@LINE+8]]:16
  // CHECK:STDERR: ->     │   let _: f32 = -a;
  // CHECK:STDERR:        ·                ┳
  // CHECK:STDERR:        ·                ┗━━┫ error: type `Cpp.ConvertibleToFloat` does not implement
  // CHECK:STDERR:        ·                   ┃        interface `Core.Negate`
  // CHECK:STDERR:        ·                   ┃        [MissingImplForType]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let _: f32 = -a;
}

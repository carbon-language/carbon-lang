// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/expressions.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/expressions.carbon

// --- test.carbon

fn F() {
  var is_valid: bool = has_permission and is_authenticated and not is_token_expired;

  // Breaks at the loosest operator (`+`, not `*`), keeps `*` groups intact,
  // and aligns wrapped operands under the first operand.
  return aaaaaaaaaa * bbbbbbbbbb + cccccccccc * dddddddddd + eeeeeeeeee * ffffffffff;

  // Keeps the first operand on the `=` line and aligns wrapped operands under
  // it when breaking after `=` would still wrap the right-hand side.
  total_value_xyz = aaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbb * ccccccccccccccc;

  // Breaks after `=` when the entire right-hand side fits on one continuation
  // line.
  total_value_xyz = aaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbb * ccccc;

  // Parentheses penalize inner breaks, so the outer `and` breaks before the
  // parenthesized `or`.
  return (first_condition_name or second_condition_name) and third_longer_condition_name_xyz;

  // Operands of an expression inside a call align after the open bracket.
  process_data(aaaaaaaaaa + bbbbbbbbbb + cccccccccc + dddddddddd + eeeeeeeeee + ffffff);

  // Bitwise `&` binds tighter than relational `==` in Carbon (`(a & b) == c`).
  return first_bitmask_value & second_mask_bits_value == expected_masked_result_value;
}

// Symbolic unary operators never split from their operands; `unsafe as` keeps
// `unsafe` attached to `as`.
fn UnaryAndCasts() -> DestinationPointerTypeNameXyz {
  TakeTheAddress(&some_rather_long_variable_name_that_makes_this_overflow_xyzzy);
  var ptr: auto = &some_extremely_long_variable_name_that_overflows_even_after_wrapping_xyzzyx;
  TakePointerType(SomeExtremelyLongParameterTypeNameGoesHereForPointerOverflow*);
  var flag: bool = not currently_enabled_setting and not alternate_mode_settings;
  return some_rather_long_raw_pointer_value unsafe as DestinationPointerTypeNameXyz;
}

impl SomeLongGenericType(i32) as SomeLongInterfaceName where .Element = AssociatedType {}

impl Vector(i32) as Printable where .Element = SomeLongAssociatedType and .Other == AnotherAssociatedType {}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   var is_valid: bool =
// CHECK:STDOUT:       has_permission and is_authenticated and not is_token_expired;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Breaks at the loosest operator (`+`, not `*`), keeps `*` groups intact,
// CHECK:STDOUT:   // and aligns wrapped operands under the first operand.
// CHECK:STDOUT:   return aaaaaaaaaa * bbbbbbbbbb + cccccccccc * dddddddddd +
// CHECK:STDOUT:          eeeeeeeeee * ffffffffff;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Keeps the first operand on the `=` line and aligns wrapped operands under
// CHECK:STDOUT:   // it when breaking after `=` would still wrap the right-hand side.
// CHECK:STDOUT:   total_value_xyz = aaaaaaaaaaaaaaaaaaaaaaaaaaa +
// CHECK:STDOUT:                     bbbbbbbbbbbbbbbbbbbbbbbbbbb * ccccccccccccccc;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Breaks after `=` when the entire right-hand side fits on one continuation
// CHECK:STDOUT:   // line.
// CHECK:STDOUT:   total_value_xyz =
// CHECK:STDOUT:       aaaaaaaaaaaaaaaaaaaaaaaaaaa + bbbbbbbbbbbbbbbbbbbbbbbbbbb * ccccc;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Parentheses penalize inner breaks, so the outer `and` breaks before the
// CHECK:STDOUT:   // parenthesized `or`.
// CHECK:STDOUT:   return (first_condition_name or second_condition_name) and
// CHECK:STDOUT:          third_longer_condition_name_xyz;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Operands of an expression inside a call align after the open bracket.
// CHECK:STDOUT:   process_data(aaaaaaaaaa + bbbbbbbbbb + cccccccccc + dddddddddd + eeeeeeeeee +
// CHECK:STDOUT:                ffffff);
// CHECK:STDOUT:
// CHECK:STDOUT:   // Bitwise `&` binds tighter than relational `==` in Carbon (`(a & b) == c`).
// CHECK:STDOUT:   return first_bitmask_value & second_mask_bits_value ==
// CHECK:STDOUT:          expected_masked_result_value;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: // Symbolic unary operators never split from their operands; `unsafe as` keeps
// CHECK:STDOUT: // `unsafe` attached to `as`.
// CHECK:STDOUT: fn UnaryAndCasts() -> DestinationPointerTypeNameXyz {
// CHECK:STDOUT:   TakeTheAddress(
// CHECK:STDOUT:       &some_rather_long_variable_name_that_makes_this_overflow_xyzzy);
// CHECK:STDOUT:   var ptr: auto =
// CHECK:STDOUT:       &some_extremely_long_variable_name_that_overflows_even_after_wrapping_xyzzyx;
// CHECK:STDOUT:   TakePointerType(
// CHECK:STDOUT:       SomeExtremelyLongParameterTypeNameGoesHereForPointerOverflow*);
// CHECK:STDOUT:   var flag: bool =
// CHECK:STDOUT:       not currently_enabled_setting and not alternate_mode_settings;
// CHECK:STDOUT:   return some_rather_long_raw_pointer_value unsafe as
// CHECK:STDOUT:          DestinationPointerTypeNameXyz;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl SomeLongGenericType(i32) as SomeLongInterfaceName
// CHECK:STDOUT:     where .Element = AssociatedType {}
// CHECK:STDOUT:
// CHECK:STDOUT: impl Vector(i32) as Printable where .Element = SomeLongAssociatedType and
// CHECK:STDOUT:                                     .Other == AnotherAssociatedType {}

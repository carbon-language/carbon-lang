// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/member_chain.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/member_chain.carbon

// --- test.carbon

fn F() {
  // A chain that fits stays on one line.
  var short: auto = obj.aaaa().bbbb().cccc();

  // A call chain too long to fit breaks before every member that follows a
  // call's `)` (the builder shape): the first segment stays attached, and the
  // rest align under the receiver at the continuation indent. The breaks are
  // all-or-nothing, so every later call breaks even though fewer would fit.
  receiverrrrrrrrrrr.aaaaaaaaaaaaaaaaa().bbbbbbbbbbbbbbbbb().cccccccccccccccc().d();

  // A pointer member-access `->` chain wraps the same way, before each `->`.
  pointerrrrrrrrrrrr->aaaaaaaaaaaaaaaa()->bbbbbbbbbbbbbbbb()->cccccccccccccc()->d();

  // The continuation aligns under the receiver, not the statement: here the
  // receiver starts after `= `, so the members indent past it (the chain is too
  // long to fit even on its own line, so breaking `=` alone would not help).
  var x: auto = aaaaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbb().cccccccccccccccc().dddddddddddddddd();

  // A field-only chain (no calls) is not a builder chain: it breaks just once,
  // before the last field, rather than at every `.`, even when the receiver is
  // parenthesized.
  some_object_receiver.field_one.field_two.field_three.field_four.field_five_long;
  (a + b).field_one.field_two.field_three.field_four.field_five.field_six_very_long_name;

  // Breaking after `=` is cheaper than breaking the chain when that lets the
  // whole chain fit on the next line, so the chain stays packed.
  var z: auto = receiver.aaaaaaaaaaaaaaaaa().bbbbbbbbbbbbbbbbb().ccccccccccccccc().d();

  // Multi-line call arguments inside a builder chain still break subsequent
  // fluent links to the receiver's continuation indent rather than packing them
  // onto the closing `)`.
  receiver.FirstCall(very_long_first_call_argument_name, very_long_second_call_argument_name).SecondCall().ThirdCall();

  // Nested builder chains and parenthesized receivers restore their respective
  // chain anchors as inner scopes close.
  outer.Call(inner_builder.FirstStep().SecondStep().ThirdStep().FourthStepWithLongName()).Build();
  (very_long_first_operand_expression + very_long_second_operand_expression).FirstStep().SecondStep().ThirdStep();

  // A builder chain followed by an infix operator stays packed when the chain
  // and attached operator fit within the column limit, wrapping after `+`.
  builder.FirstStep().SecondStep() + very_long_right_hand_side_operand_expression;
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   // A chain that fits stays on one line.
// CHECK:STDOUT:   var short: auto = obj.aaaa().bbbb().cccc();
// CHECK:STDOUT:
// CHECK:STDOUT:   // A call chain too long to fit breaks before every member that follows a
// CHECK:STDOUT:   // call's `)` (the builder shape): the first segment stays attached, and the
// CHECK:STDOUT:   // rest align under the receiver at the continuation indent. The breaks are
// CHECK:STDOUT:   // all-or-nothing, so every later call breaks even though fewer would fit.
// CHECK:STDOUT:   receiverrrrrrrrrrr.aaaaaaaaaaaaaaaaa()
// CHECK:STDOUT:       .bbbbbbbbbbbbbbbbb()
// CHECK:STDOUT:       .cccccccccccccccc()
// CHECK:STDOUT:       .d();
// CHECK:STDOUT:
// CHECK:STDOUT:   // A pointer member-access `->` chain wraps the same way, before each `->`.
// CHECK:STDOUT:   pointerrrrrrrrrrrr->aaaaaaaaaaaaaaaa()
// CHECK:STDOUT:       ->bbbbbbbbbbbbbbbb()
// CHECK:STDOUT:       ->cccccccccccccc()
// CHECK:STDOUT:       ->d();
// CHECK:STDOUT:
// CHECK:STDOUT:   // The continuation aligns under the receiver, not the statement: here the
// CHECK:STDOUT:   // receiver starts after `= `, so the members indent past it (the chain is too
// CHECK:STDOUT:   // long to fit even on its own line, so breaking `=` alone would not help).
// CHECK:STDOUT:   var x: auto = aaaaaaaaaaaaaaaaaaaaaaaa.bbbbbbbbbbbbbbbb()
// CHECK:STDOUT:                     .cccccccccccccccc()
// CHECK:STDOUT:                     .dddddddddddddddd();
// CHECK:STDOUT:
// CHECK:STDOUT:   // A field-only chain (no calls) is not a builder chain: it breaks just once,
// CHECK:STDOUT:   // before the last field, rather than at every `.`, even when the receiver is
// CHECK:STDOUT:   // parenthesized.
// CHECK:STDOUT:   some_object_receiver.field_one.field_two.field_three.field_four
// CHECK:STDOUT:       .field_five_long;
// CHECK:STDOUT:   (a + b).field_one.field_two.field_three.field_four.field_five
// CHECK:STDOUT:       .field_six_very_long_name;
// CHECK:STDOUT:
// CHECK:STDOUT:   // Breaking after `=` is cheaper than breaking the chain when that lets the
// CHECK:STDOUT:   // whole chain fit on the next line, so the chain stays packed.
// CHECK:STDOUT:   var z: auto =
// CHECK:STDOUT:       receiver.aaaaaaaaaaaaaaaaa().bbbbbbbbbbbbbbbbb().ccccccccccccccc().d();
// CHECK:STDOUT:
// CHECK:STDOUT:   // Multi-line call arguments inside a builder chain still break subsequent
// CHECK:STDOUT:   // fluent links to the receiver's continuation indent rather than packing them
// CHECK:STDOUT:   // onto the closing `)`.
// CHECK:STDOUT:   receiver.FirstCall(very_long_first_call_argument_name,
// CHECK:STDOUT:                      very_long_second_call_argument_name)
// CHECK:STDOUT:       .SecondCall()
// CHECK:STDOUT:       .ThirdCall();
// CHECK:STDOUT:
// CHECK:STDOUT:   // Nested builder chains and parenthesized receivers restore their respective
// CHECK:STDOUT:   // chain anchors as inner scopes close.
// CHECK:STDOUT:   outer.Call(inner_builder.FirstStep()
// CHECK:STDOUT:                  .SecondStep()
// CHECK:STDOUT:                  .ThirdStep()
// CHECK:STDOUT:                  .FourthStepWithLongName())
// CHECK:STDOUT:       .Build();
// CHECK:STDOUT:   (very_long_first_operand_expression +
// CHECK:STDOUT:    very_long_second_operand_expression).FirstStep()
// CHECK:STDOUT:       .SecondStep()
// CHECK:STDOUT:       .ThirdStep();
// CHECK:STDOUT:
// CHECK:STDOUT:   // A builder chain followed by an infix operator stays packed when the chain
// CHECK:STDOUT:   // and attached operator fit within the column limit, wrapping after `+`.
// CHECK:STDOUT:   builder.FirstStep().SecondStep() +
// CHECK:STDOUT:       very_long_right_hand_side_operand_expression;
// CHECK:STDOUT: }

// RUN: not llvm-mc -arch thumb -mattr=+thumb2 \
// RUN: < %s >/dev/null 2> %t
// RUN: grep "error: destination operands can't be identical" %t | count 4
// rdar://14479780

ldrd r0, r0, [pc, #0]
ldrd r0, r0, [r1, #4]
ldrd r0, r0, [r1], #4
ldrd r0, r0, [r1, #4]!

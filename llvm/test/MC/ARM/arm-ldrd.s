// RUN: not llvm-mc -arch arm -mattr=+v5te \
// RUN: < %s >/dev/null 2> %t
// RUN: grep "error: Rt must be even-numbered" %t | count 7
// RUN: grep "error: Rt can't be R14" %t | count 7
// RUN: grep "error: destination operands must be sequential" %t | count 7
// RUN: grep "error: base register needs to be different from destination registers" %t | count 4
// rdar://14479793

ldrd r1, r2, [pc, #0]
ldrd lr, pc, [pc, #0]
ldrd r0, r3, [pc, #0]
ldrd r1, r2, [r3, #4]
ldrd lr, pc, [r3, #4]
ldrd r0, r3, [r4, #4]
ldrd r1, r2, [r3], #4
ldrd lr, pc, [r3], #4
ldrd r0, r3, [r4], #4
ldrd r1, r2, [r3, #4]!
ldrd lr, pc, [r3, #4]!
ldrd r0, r3, [r4, #4]!
ldrd r1, r2, [r3, -r4]!
ldrd lr, pc, [r3, -r4]!
ldrd r0, r3, [r4, -r5]!
ldrd r1, r2, [r3, r4]
ldrd lr, pc, [r3, r4]
ldrd r0, r3, [r4, r5]
ldrd r1, r2, [r3], r4
ldrd lr, pc, [r3], r4
ldrd r0, r3, [r4], r5

ldrd r0, r1, [r0], #4
ldrd r0, r1, [r1], #4
ldrd r0, r1, [r0, #4]!
ldrd r0, r1, [r1, #4]!

// RUN: not llvm-mc -arch arm -mattr=+v5te \
// RUN: < %s >/dev/null 2> %t
// RUN: grep "error: Rt must be even-numbered" %t | count 7
// RUN: grep "error: Rt can't be R14" %t | count 7
// rdar://14479793

ldrd r1, r2, [pc, #0]
ldrd lr, pc, [pc, #0]
ldrd r1, r2, [r3, #4]
ldrd lr, pc, [r3, #4]
ldrd r1, r2, [r3], #4
ldrd lr, pc, [r3], #4
ldrd r1, r2, [r3, #4]!
ldrd lr, pc, [r3, #4]!
ldrd r1, r2, [r3, -r4]!
ldrd lr, pc, [r3, -r4]!
ldrd r1, r2, [r3, r4]
ldrd lr, pc, [r3, r4]
ldrd r1, r2, [r3], r4
ldrd lr, pc, [r3], r4

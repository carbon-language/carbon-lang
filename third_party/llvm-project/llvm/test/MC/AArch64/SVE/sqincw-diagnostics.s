// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

sqincw w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincw w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw wsp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincw wsp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincw sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw z0.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: sqincw z0.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Operands not matching up

sqincw x0, w1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must be 32-bit form of destination register
// CHECK-NEXT: sqincw x0, w1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw x0, x0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincw x0, x0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

sqincw x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqincw x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqincw x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: sqincw x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

sqincw x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincw x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw x0, vl9
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: sqincw x0, vl9
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqincw x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

sqincw x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: sqincw x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.s, p0/z, z7.s
sqincw  z0.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqincw  z0.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
sqincw  z0.s, pow2, mul #16
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqincw  z0.s, pow2, mul #16
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0.s, p0/z, z7.s
sqincw  z0.s, pow2
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: sqincw  z0.s, pow2
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

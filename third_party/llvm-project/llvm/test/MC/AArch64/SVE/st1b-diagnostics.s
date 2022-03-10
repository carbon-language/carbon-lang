// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Immediate out of upper bound [-8, 7].

st1b z10.b, p4, [x8, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z10.b, p4, [x8, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z18.b, p4, [x24, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z18.b, p4, [x24, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z11.h, p0, [x23, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z11.h, p0, [x23, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z24.h, p3, [x1, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z24.h, p3, [x1, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z6.s, p5, [x23, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z6.s, p5, [x23, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z16.s, p6, [x14, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z16.s, p6, [x14, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z26.d, p2, [x7, #-9, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z26.d, p2, [x7, #-9, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z27.d, p1, [x12, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: index must be an integer in range [-8, 7].
// CHECK-NEXT: st1b z27.d, p1, [x12, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid predicate

st1b z12.b, p8, [x27, #6, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st1b z12.b, p8, [x27, #6, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z23.h, p8, [x20, #1, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st1b z23.h, p8, [x20, #1, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z6.s, p8, [x0, #8, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st1b z6.s, p8, [x0, #8, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z14.d, p8, [x6, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st1b z14.d, p8, [x6, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z14.d, p7.b, [x6, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st1b z14.d, p7.b, [x6, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z14.d, p7.q, [x6, #5, MUL VL]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: st1b z14.d, p7.q, [x6, #5, MUL VL]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// --------------------------------------------------------------------------//
// Invalid vector list

st1b { }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector register expected
// CHECK-NEXT: st1b { }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b { z1.b, z2.b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1b { z1.b, z2.b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b { v0.16b }, p0, [x0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1b { v0.16b }, p0, [x0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + scalar addressing modes

st1b z0.b, p0, [x0, xzr]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, xzr]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.b, p0, [x0, x0, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, x0, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.b, p0, [x0, w0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, w0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.b, p0, [x0, w0, uxtw]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: register must be x0..x30 without shift
// CHECK-NEXT: st1b z0.b, p0, [x0, w0, uxtw]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid scalar + vector addressing modes

st1b z0.d, p0, [x0, z0.b]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1b z0.d, p0, [x0, z0.b]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.d, p0, [x0, z0.h]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1b z0.d, p0, [x0, z0.h]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.d, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: st1b z0.d, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.s, p0, [x0, z0.s]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw)'
// CHECK-NEXT: st1b z0.s, p0, [x0, z0.s]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.s, p0, [x0, z0.s, uxtw #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw)'
// CHECK-NEXT: st1b z0.s, p0, [x0, z0.s, uxtw #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.s, p0, [x0, z0.s, lsl #0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].s, (uxtw|sxtw)'
// CHECK-NEXT: st1b z0.s, p0, [x0, z0.s, lsl #0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.d, p0, [x0, z0.d, lsl #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (uxtw|sxtw)'
// CHECK-NEXT: st1b z0.d, p0, [x0, z0.d, lsl #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.d, p0, [x0, z0.d, sxtw #1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid shift/extend specified, expected 'z[0..31].d, (uxtw|sxtw)'
// CHECK-NEXT: st1b z0.d, p0, [x0, z0.d, sxtw #1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Invalid vector + immediate addressing modes

st1b z0.s, p0, [z0.s, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31].
// CHECK-NEXT: st1b z0.s, p0, [z0.s, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.s, p0, [z0.s, #32]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31].
// CHECK-NEXT: st1b z0.s, p0, [z0.s, #32]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.d, p0, [z0.d, #-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31].
// CHECK-NEXT: st1b z0.d, p0, [z0.d, #-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

st1b z0.d, p0, [z0.d, #32]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [0, 31].
// CHECK-NEXT: st1b z0.d, p0, [z0.d, #32]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
st1b    { z31.d }, p7, [z31.d, #31]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st1b    { z31.d }, p7, [z31.d, #31]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
st1b    { z31.d }, p7, [z31.d, #31]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: st1b    { z31.d }, p7, [z31.d, #31]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

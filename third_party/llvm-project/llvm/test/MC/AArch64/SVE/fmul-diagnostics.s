// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// --------------------------------------------------------------------------//
// Invalid immediates (must be 0.5 or 2.0)

fmul z0.h, p0/m, z0.h, #1.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 2.0.
// CHECK-NEXT: fmul z0.h, p0/m, z0.h, #1.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, p0/m, z0.h, #0.0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 2.0.
// CHECK-NEXT: fmul z0.h, p0/m, z0.h, #0.0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, p0/m, z0.h, #0.4999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 2.0.
// CHECK-NEXT: fmul z0.h, p0/m, z0.h, #0.4999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, p0/m, z0.h, #0.5000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 2.0.
// CHECK-NEXT: fmul z0.h, p0/m, z0.h, #0.5000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, p0/m, z0.h, #2.0000000000000000000000001
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 2.0.
// CHECK-NEXT: fmul z0.h, p0/m, z0.h, #2.0000000000000000000000001
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, p0/m, z0.h, #1.9999999999999999999999999
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid floating point constant, expected 0.5 or 2.0.
// CHECK-NEXT: fmul z0.h, p0/m, z0.h, #1.9999999999999999999999999
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Restricted ZPR range

fmul z0.h, z0.h, z8.b[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: Invalid restricted vector register, expected z0.h..z7.h
// CHECK-NEXT: fmul z0.h, z0.h, z8.b[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, z0.h, z8.h[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmul z0.h, z0.h, z8.h[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.s, z0.s, z8.s[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmul z0.s, z0.s, z8.s[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.d, z0.d, z16.d[0]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fmul z0.d, z0.d, z16.d[0]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Index out of bounds

fmul z0.h, z0.h, z0.h[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fmul z0.h, z0.h, z0.h[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, z0.h, z0.h[8]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 7].
// CHECK-NEXT: fmul z0.h, z0.h, z0.h[8]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.s, z0.s, z0.s[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fmul z0.s, z0.s, z0.s[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.s, z0.s, z0.s[4]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 3].
// CHECK-NEXT: fmul z0.s, z0.s, z0.s[4]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.d, z0.d, z0.d[-1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: fmul z0.d, z0.d, z0.d[-1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.d, z0.d, z0.d[2]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: vector lane must be an integer in range [0, 1].
// CHECK-NEXT: fmul z0.d, z0.d, z0.d[2]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Tied operands must match

fmul    z0.h, p7/m, z1.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: fmul    z0.h, p7/m, z1.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid element widths.

fmul    z0.b, p7/m, z0.b, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmul    z0.b, p7/m, z0.b, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul    z0.h, p7/m, z0.h, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmul    z0.h, p7/m, z0.h, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.b, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmul z0.b, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fmul z0.h, z1.s, z2.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: fmul z0.h, z1.s, z2.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

fmul    z0.h, p8/m, z0.h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fmul    z0.h, p8/m, z0.h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
fmul z0.d, z1.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fmul z0.d, z1.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z0, z7
fmul z0.d, z1.d, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fmul z0.d, z1.d, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31.d, p0/z, z6.d
fmul    z31.d, z31.d, z15.d[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fmul    z31.d, z31.d, z15.d[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
fmul    z31.d, z31.d, z15.d[1]
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fmul    z31.d, z31.d, z15.d[1]
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

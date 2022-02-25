// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

fadda b0, p7, b0, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fadda b0, p7, b0, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadda h0, p7, h1, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: fadda h0, p7, h1, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadda v0.8h, p7, v0.8h, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: fadda v0.8h, p7, v0.8h, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate operand

fadda h0, p8, h0, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fadda h0, p8, h0, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadda h0, p7.b, h0, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fadda h0, p7.b, h0, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

fadda h0, p7.q, h0, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: fadda h0, p7.q, h0, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
fadda d0, p7, d0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fadda d0, p7, d0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
fadda d0, p7, d0, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: fadda d0, p7, d0, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

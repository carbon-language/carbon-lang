// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid destination or source register.

umaxv d0, p7, z31.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: umaxv d0, p7, z31.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umaxv d0, p7, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: umaxv d0, p7, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umaxv d0, p7, z31.s
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid element width
// CHECK-NEXT: umaxv d0, p7, z31.s
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umaxv v0.2d, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: umaxv v0.2d, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

umaxv h0, p8, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: umaxv h0, p8, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umaxv h0, p7.b, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: umaxv h0, p7.b, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

umaxv h0, p7.q, z31.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: umaxv h0, p7.q, z31.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z31.d, p7/z, z6.d
umaxv d0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: umaxv d0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

movprfx z31, z6
umaxv d0, p7, z31.d
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a movprfx, suggest replacing movprfx with mov
// CHECK-NEXT: umaxv d0, p7, z31.d
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

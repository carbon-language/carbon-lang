// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Tied operands must match

smulh z0.b, p7/m, z1.b, z2.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: smulh z0.b, p7/m, z1.b, z2.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate

smulh z0.b, p8/m, z0.b, z1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid restricted predicate register, expected p0..p7 (without element suffix)
// CHECK-NEXT: smulh z0.b, p8/m, z0.b, z1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

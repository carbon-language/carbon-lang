// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s


// ------------------------------------------------------------------------- //
// Invalid predicate

pfirst p0.h, p15, p0.h
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: pfirst p0.h, p15, p0.h
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pfirst p0.b, p15/z, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: pfirst p0.b, p15/z, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pfirst p0.b, p15/m, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand for instruction
// CHECK-NEXT: pfirst p0.b, p15/m, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pfirst p0.b, p15.b, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: pfirst p0.b, p15.b, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

pfirst p0.b, p15.q, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: pfirst p0.b, p15.q, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Tied operands must match

pfirst p0.b, p15, p1.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: operand must match destination register
// CHECK-NEXT: pfirst p0.b, p15, p1.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+sve  2>&1 < %s| FileCheck %s

// ------------------------------------------------------------------------- //
// Invalid result register

cntw  w0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntw  w0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntw  sp
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntw  sp
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntw  z0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntw  z0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Immediate not compatible with encode/decode function.

cntw  x0, all, mul #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cntw  x0, all, mul #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntw  x0, all, mul #0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cntw  x0, all, mul #0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntw  x0, all, mul #17
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: immediate must be an integer in range [1, 16]
// CHECK-NEXT: cntw  x0, all, mul #17
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// ------------------------------------------------------------------------- //
// Invalid predicate patterns

cntw  x0, #-1
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: cntw  x0, #-1
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntw  x0, #32
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate pattern
// CHECK-NEXT: cntw  x0, #32
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

cntw  x0, vl512
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid operand
// CHECK-NEXT: cntw  x0, vl512
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

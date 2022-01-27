// RUN: not llvm-mc -triple=aarch64-none-linux-gnu -show-encoding -mattr=+sve  2>&1 < %s | FileCheck %s

uqdecp z0.d, p0.b
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: uqdecp z0.d, p0.b
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

uqdecp z0.d, p0.q
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: invalid predicate register
// CHECK-NEXT: uqdecp z0.d, p0.q
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:


// --------------------------------------------------------------------------//
// Negative tests for instructions that are incompatible with movprfx

movprfx z0.d, p0/z, z7.d
uqdecp  z0.d, p0
// CHECK: [[@LINE-1]]:{{[0-9]+}}: error: instruction is unpredictable when following a predicated movprfx, suggest using unpredicated movprfx
// CHECK-NEXT: uqdecp  z0.d, p0
// CHECK-NOT: [[@LINE-1]]:{{[0-9]+}}:

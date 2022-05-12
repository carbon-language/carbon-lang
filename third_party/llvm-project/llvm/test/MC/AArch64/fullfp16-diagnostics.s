// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon < %s 2> %t
// RUN: FileCheck < %t %s

  fmla v0.4h, v1.4h, v16.h[3]
  fmla v2.8h, v3.8h, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmla v0.4h, v1.4h, v16.h[3]
// CHECK-NEXT:                    ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmla v2.8h, v3.8h, v17.h[6]
// CHECK-NEXT:                    ^

  fmls v0.4h, v1.4h, v16.h[3]
  fmls v2.8h, v3.8h, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmls v0.4h, v1.4h, v16.h[3]
// CHECK-NEXT:                    ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmls v2.8h, v3.8h, v17.h[6]
// CHECK-NEXT:                    ^

  fmul v0.4h, v1.4h, v16.h[3]
  fmul v2.8h, v3.8h, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmul v0.4h, v1.4h, v16.h[3]
// CHECK-NEXT:                    ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmul v2.8h, v3.8h, v17.h[6]
// CHECK-NEXT:                    ^

  fmulx v0.4h, v1.4h, v16.h[3]
  fmulx v2.8h, v3.8h, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmulx v0.4h, v1.4h, v16.h[3]
// CHECK-NEXT:                     ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmulx v2.8h, v3.8h, v17.h[6]
// CHECK-NEXT:                     ^

  fmla h0, h1, v16.h[3]
  fmla h2, h3, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmla h0, h1, v16.h[3]
// CHECK-NEXT:              ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmla h2, h3, v17.h[6]
// CHECK-NEXT:              ^

  fmls h0, h1, v16.h[3]
  fmls h2, h3, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmls h0, h1, v16.h[3]
// CHECK-NEXT:              ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmls h2, h3, v17.h[6]
// CHECK-NEXT:              ^

  fmul h0, h1, v16.h[3]
  fmul h2, h3, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmul h0, h1, v16.h[3]
// CHECK-NEXT:              ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmul h2, h3, v17.h[6]
// CHECK-NEXT:              ^

  fmulx h0, h1, v16.h[3]
  fmulx h2, h3, v17.h[6]

// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmulx h0, h1, v16.h[3]
// CHECK-NEXT:               ^
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: fmulx h2, h3, v17.h[6]
// CHECK-NEXT:               ^

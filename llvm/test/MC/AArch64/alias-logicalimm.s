// RUN: llvm-mc -triple=aarch64-none-linux-gnu < %s | FileCheck %s

// CHECK: and x0, x1, #0xfffffffffffffffd
// CHECK: and x0, x1, #0xfffffffffffffffd
        and x0, x1, #~2
        bic x0, x1, #2

// CHECK: and w0, w1, #0xfffffffd
// CHECK: and w0, w1, #0xfffffffd
        and w0, w1, #~2
        bic w0, w1, #2

// CHECK: ands x0, x1, #0xfffffffffffffffd
// CHECK: ands x0, x1, #0xfffffffffffffffd
        ands x0, x1, #~2
        bics x0, x1, #2

// CHECK: ands w0, w1, #0xfffffffd
// CHECK: ands w0, w1, #0xfffffffd
        ands w0, w1, #~2
        bics w0, w1, #2

// CHECK: orr x0, x1, #0xfffffffffffffffd
// CHECK: orr x0, x1, #0xfffffffffffffffd
        orr x0, x1, #~2
        orn x0, x1, #2

// CHECK: orr w2, w1, #0xfffffffc
// CHECK: orr w2, w1, #0xfffffffc
        orr w2, w1, #~3
        orn w2, w1, #3

// CHECK: eor x0, x1, #0xfffffffffffffffd
// CHECK: eor x0, x1, #0xfffffffffffffffd
        eor x0, x1, #~2
        eon x0, x1, #2

// CHECK: eor w2, w1, #0xfffffffc
// CHECK: eor w2, w1, #0xfffffffc
        eor w2, w1, #~3
        eon w2, w1, #3

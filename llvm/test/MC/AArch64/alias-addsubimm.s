// RUN: llvm-mc -triple=aarch64-none-linux-gnu < %s | FileCheck %s

// CHECK: sub w0, w2, #2, lsl #12
// CHECK: sub w0, w2, #2, lsl #12
        sub w0, w2, #2, lsl 12
        add w0, w2, #-2, lsl 12
// CHECK: sub x1, x3, #2, lsl #12
// CHECK: sub x1, x3, #2, lsl #12
        sub x1, x3, #2, lsl 12
        add x1, x3, #-2, lsl 12
// CHECK: sub x1, x3, #4
// CHECK: sub x1, x3, #4
        sub x1, x3, #4
        add x1, x3, #-4
// CHECK: sub x1, x3, #4095
// CHECK: sub x1, x3, #4095
        sub x1, x3, #4095, lsl 0
        add x1, x3, #-4095, lsl 0
// CHECK: sub x3, x4, #0
        sub x3, x4, #0

// CHECK: add w0, w2, #2, lsl #12
// CHECK: add w0, w2, #2, lsl #12
        add w0, w2, #2, lsl 12
        sub w0, w2, #-2, lsl 12
// CHECK: add x1, x3, #2, lsl #12
// CHECK: add x1, x3, #2, lsl #12
        add x1, x3, #2, lsl 12
        sub x1, x3, #-2, lsl 12
// CHECK: add x1, x3, #4
// CHECK: add x1, x3, #4
        add x1, x3, #4
        sub x1, x3, #-4
// CHECK: add x1, x3, #4095
// CHECK: add x1, x3, #4095
        add x1, x3, #4095, lsl 0
        sub x1, x3, #-4095, lsl 0
// CHECK: add x2, x5, #0
        add x2, x5, #0

// CHECK: subs w0, w2, #2, lsl #12
// CHECK: subs w0, w2, #2, lsl #12
        subs w0, w2, #2, lsl 12
        adds w0, w2, #-2, lsl 12
// CHECK: subs x1, x3, #2, lsl #12
// CHECK: subs x1, x3, #2, lsl #12
        subs x1, x3, #2, lsl 12
        adds x1, x3, #-2, lsl 12
// CHECK: subs x1, x3, #4
// CHECK: subs x1, x3, #4
        subs x1, x3, #4
        adds x1, x3, #-4
// CHECK: subs x1, x3, #4095
// CHECK: subs x1, x3, #4095
        subs x1, x3, #4095, lsl 0
        adds x1, x3, #-4095, lsl 0
// CHECK: subs x3, x4, #0
        subs x3, x4, #0

// CHECK: adds w0, w2, #2, lsl #12
// CHECK: adds w0, w2, #2, lsl #12
        adds w0, w2, #2, lsl 12
        subs w0, w2, #-2, lsl 12
// CHECK: adds x1, x3, #2, lsl #12
// CHECK: adds x1, x3, #2, lsl #12
        adds x1, x3, #2, lsl 12
        subs x1, x3, #-2, lsl 12
// CHECK: adds x1, x3, #4
// CHECK: adds x1, x3, #4
        adds x1, x3, #4
        subs x1, x3, #-4
// CHECK: adds x1, x3, #4095
// CHECK: adds x1, x3, #4095
        adds x1, x3, #4095, lsl 0
        subs x1, x3, #-4095, lsl 0
// CHECK: adds x2, x5, #0
        adds x2, x5, #0

// CHECK: {{adds xzr,|cmn}} x5, #5
// CHECK: {{adds xzr,|cmn}} x5, #5
        cmn x5, #5
        cmp x5, #-5
// CHECK: {{subs xzr,|cmp}} x6, #4095
// CHECK: {{subs xzr,|cmp}} x6, #4095
        cmp x6, #4095
        cmn x6, #-4095
// CHECK: {{adds wzr,|cmn}} w7, #5
// CHECK: {{adds wzr,|cmn}} w7, #5
        cmn w7, #5
        cmp w7, #-5
// CHECK: {{subs wzr,|cmp}} w8, #4095
// CHECK: {{subs wzr,|cmp}} w8, #4095
        cmp w8, #4095
        cmn w8, #-4095

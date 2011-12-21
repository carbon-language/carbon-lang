@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

@ CHECK: vadd.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0x71,0xee]
        vadd.f64        d16, d17, d16

@ CHECK: vadd.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x30,0xee]
        vadd.f32        s0, s1, s0

@ CHECK: vsub.f64 d16, d17, d16      @ encoding: [0xe0,0x0b,0x71,0xee]
        vsub.f64        d16, d17, d16

@ CHECK: vsub.f32 s0, s1, s0         @ encoding: [0xc0,0x0a,0x30,0xee]
        vsub.f32        s0, s1, s0

@ CHECK: vdiv.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0xc1,0xee]
        vdiv.f64        d16, d17, d16

@ CHECK: vdiv.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x80,0xee]
        vdiv.f32        s0, s1, s0

@ CHECK: vmul.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0x61,0xee]
        vmul.f64        d16, d17, d16

@ CHECK: vmul.f64	d20, d20, d17   @ encoding: [0xa1,0x4b,0x64,0xee]
	vmul.f64  d20, d17

@ CHECK: vmul.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x20,0xee]
        vmul.f32        s0, s1, s0

@ CHECK: vmul.f32	s11, s11, s21   @ encoding: [0xaa,0x5a,0x65,0xee]
	vmul.f32  s11, s21

@ CHECK: vnmul.f64 d16, d17, d16     @ encoding: [0xe0,0x0b,0x61,0xee]
        vnmul.f64       d16, d17, d16

@ CHECK: vnmul.f32 s0, s1, s0        @ encoding: [0xc0,0x0a,0x20,0xee]
        vnmul.f32       s0, s1, s0

@ CHECK: vcmpe.f64 d17, d16          @ encoding: [0xe0,0x1b,0xf4,0xee]
        vcmpe.f64       d17, d16

@ CHECK: vcmpe.f32 s1, s0            @ encoding: [0xc0,0x0a,0xf4,0xee]
        vcmpe.f32       s1, s0

@ CHECK: vcmpe.f64 d16, #0           @ encoding: [0xc0,0x0b,0xf5,0xee]
        vcmpe.f64       d16, #0

@ CHECK: vcmpe.f32 s0, #0            @ encoding: [0xc0,0x0a,0xb5,0xee]
        vcmpe.f32       s0, #0

@ CHECK: vabs.f64 d16, d16           @ encoding: [0xe0,0x0b,0xf0,0xee]
        vabs.f64        d16, d16

@ CHECK: vabs.f32 s0, s0             @ encoding: [0xc0,0x0a,0xb0,0xee]
        vabs.f32        s0, s0

@ CHECK: vcvt.f32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xb7,0xee]
        vcvt.f32.f64    s0, d16

@ CHECK: vcvt.f64.f32 d16, s0        @ encoding: [0xc0,0x0a,0xf7,0xee]
        vcvt.f64.f32    d16, s0

@ CHECK: vneg.f64 d16, d16           @ encoding: [0x60,0x0b,0xf1,0xee]
        vneg.f64        d16, d16

@ CHECK: vneg.f32 s0, s0             @ encoding: [0x40,0x0a,0xb1,0xee]
        vneg.f32        s0, s0

@ CHECK: vsqrt.f64 d16, d16          @ encoding: [0xe0,0x0b,0xf1,0xee]
        vsqrt.f64       d16, d16

@ CHECK: vsqrt.f32 s0, s0            @ encoding: [0xc0,0x0a,0xb1,0xee]
        vsqrt.f32       s0, s0

@ CHECK: vcvt.f64.s32 d16, s0        @ encoding: [0xc0,0x0b,0xf8,0xee]
        vcvt.f64.s32    d16, s0

@ CHECK: vcvt.f32.s32 s0, s0         @ encoding: [0xc0,0x0a,0xb8,0xee]
        vcvt.f32.s32    s0, s0

@ CHECK: vcvt.f64.u32 d16, s0        @ encoding: [0x40,0x0b,0xf8,0xee]
        vcvt.f64.u32    d16, s0

@ CHECK: vcvt.f32.u32 s0, s0         @ encoding: [0x40,0x0a,0xb8,0xee]
        vcvt.f32.u32    s0, s0

@ CHECK: vcvt.s32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xbd,0xee]
        vcvt.s32.f64    s0, d16

@ CHECK: vcvt.s32.f32 s0, s0         @ encoding: [0xc0,0x0a,0xbd,0xee]
        vcvt.s32.f32    s0, s0

@ CHECK: vcvt.u32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xbc,0xee]
        vcvt.u32.f64    s0, d16

@ CHECK: vcvt.u32.f32 s0, s0         @ encoding: [0xc0,0x0a,0xbc,0xee]
        vcvt.u32.f32    s0, s0

@ CHECK: vmla.f64 d16, d18, d17      @ encoding: [0xa1,0x0b,0x42,0xee]
        vmla.f64        d16, d18, d17

@ CHECK: vmla.f32 s1, s2, s0         @ encoding: [0x00,0x0a,0x41,0xee]
        vmla.f32        s1, s2, s0

@ CHECK: vmls.f64 d16, d18, d17      @ encoding: [0xe1,0x0b,0x42,0xee]
        vmls.f64        d16, d18, d17

@ CHECK: vmls.f32 s1, s2, s0         @ encoding: [0x40,0x0a,0x41,0xee]
        vmls.f32        s1, s2, s0

@ CHECK: vnmla.f64 d16, d18, d17     @ encoding: [0xe1,0x0b,0x52,0xee]
        vnmla.f64       d16, d18, d17

@ CHECK: vnmla.f32 s1, s2, s0        @ encoding: [0x40,0x0a,0x51,0xee]
        vnmla.f32       s1, s2, s0

@ CHECK: vnmls.f64 d16, d18, d17     @ encoding: [0xa1,0x0b,0x52,0xee]
        vnmls.f64       d16, d18, d17

@ CHECK: vnmls.f32 s1, s2, s0        @ encoding: [0x00,0x0a,0x51,0xee]
        vnmls.f32       s1, s2, s0

@ CHECK: vmrs apsr_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
@ CHECK: vmrs apsr_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
        vmrs    apsr_nzcv, fpscr
        fmstat

@ CHECK: vnegne.f64 d16, d16         @ encoding: [0x60,0x0b,0xf1,0x1e]
        vnegne.f64      d16, d16

@ CHECK: vmovne s0, r0               @ encoding: [0x10,0x0a,0x00,0x1e]
@ CHECK: vmoveq s0, r1               @ encoding: [0x10,0x1a,0x00,0x0e]
        vmovne  s0, r0
        vmoveq  s0, r1

        vmov.f32 r1, s2
        vmov.f32 s4, r3
        vmov.f64 r1, r5, d2
        vmov.f64 d4, r3, r9

@ CHECK: vmov	r1, s2                  @ encoding: [0x10,0x1a,0x11,0xee]
@ CHECK: vmov	s4, r3                  @ encoding: [0x10,0x3a,0x02,0xee]
@ CHECK: vmov	r1, r5, d2              @ encoding: [0x12,0x1b,0x55,0xec]
@ CHECK: vmov	d4, r3, r9              @ encoding: [0x14,0x3b,0x49,0xec]

@ CHECK: vmrs r0, fpscr              @ encoding: [0x10,0x0a,0xf1,0xee]
        vmrs    r0, fpscr
@ CHECK: vmrs  r0, fpexc             @ encoding: [0x10,0x0a,0xf8,0xee]
        vmrs  r0, fpexc
@ CHECK: vmrs  r0, fpsid             @ encoding: [0x10,0x0a,0xf0,0xee]
        vmrs  r0, fpsid

@ CHECK: vmsr fpscr, r0              @ encoding: [0x10,0x0a,0xe1,0xee]
        vmsr    fpscr, r0
@ CHECK: vmsr  fpexc, r0             @ encoding: [0x10,0x0a,0xe8,0xee]
        vmsr  fpexc, r0
@ CHECK: vmsr  fpsid, r0             @ encoding: [0x10,0x0a,0xe0,0xee]
        vmsr  fpsid, r0

        vmov.f64        d16, #3.000000e+00
        vmov.f32        s0, #3.000000e+00
        vmov.f64        d16, #-3.000000e+00
        vmov.f32        s0, #-3.000000e+00

@ CHECK: vmov.f64 d16, #3.000000e+00 @ encoding: [0x08,0x0b,0xf0,0xee]
@ CHECK: vmov.f32 s0, #3.000000e+00  @ encoding: [0x08,0x0a,0xb0,0xee]
@ CHECK: vmov.f64 d16, #-3.000000e+00 @ encoding: [0x08,0x0b,0xf8,0xee]
@ CHECK: vmov.f32 s0, #-3.000000e+00  @ encoding: [0x08,0x0a,0xb8,0xee]

@ CHECK: vmov s0, r0                 @ encoding: [0x10,0x0a,0x00,0xee]
@ CHECK: vmov s1, r1                 @ encoding: [0x90,0x1a,0x00,0xee]
@ CHECK: vmov s2, r2                 @ encoding: [0x10,0x2a,0x01,0xee]
@ CHECK: vmov s3, r3                 @ encoding: [0x90,0x3a,0x01,0xee]
        vmov    s0, r0
        vmov    s1, r1
        vmov    s2, r2
        vmov    s3, r3

@ CHECK: vmov r0, s0                 @ encoding: [0x10,0x0a,0x10,0xee]
@ CHECK: vmov r1, s1                 @ encoding: [0x90,0x1a,0x10,0xee]
@ CHECK: vmov r2, s2                 @ encoding: [0x10,0x2a,0x11,0xee]
@ CHECK: vmov r3, s3                 @ encoding: [0x90,0x3a,0x11,0xee]
        vmov    r0, s0
        vmov    r1, s1
        vmov    r2, s2
        vmov    r3, s3

@ CHECK: vmov r0, r1, d16            @ encoding: [0x30,0x0b,0x51,0xec]
        vmov    r0, r1, d16

@ CHECK: vldr d17, [r0]           @ encoding: [0x00,0x1b,0xd0,0xed]
@ CHECK: vldr s0, [lr]            @ encoding: [0x00,0x0a,0x9e,0xed]
@ CHECK: vldr d0, [lr]            @ encoding: [0x00,0x0b,0x9e,0xed]

        vldr.64	d17, [r0]
	vldr.i32 s0, [lr]
	vldr.d d0, [lr]

@ CHECK: vldr d1, [r2, #32]       @ encoding: [0x08,0x1b,0x92,0xed]
@ CHECK: vldr d1, [r2, #-32]      @ encoding: [0x08,0x1b,0x12,0xed]
        vldr.64	d1, [r2, #32]
        vldr.f64	d1, [r2, #-32]

@ CHECK: vldr d2, [r3]            @ encoding: [0x00,0x2b,0x93,0xed]
        vldr.64 d2, [r3]

@ CHECK: vldr d3, [pc]            @ encoding: [0x00,0x3b,0x9f,0xed]
@ CHECK: vldr d3, [pc]            @ encoding: [0x00,0x3b,0x9f,0xed]
@ CHECK: vldr d3, [pc, #-0]            @ encoding: [0x00,0x3b,0x1f,0xed]
        vldr.64 d3, [pc]
        vldr.64 d3, [pc,#0]
        vldr.64 d3, [pc,#-0]

@ CHECK: vldr s13, [r0]           @ encoding: [0x00,0x6a,0xd0,0xed]
        vldr.32	s13, [r0]

@ CHECK: vldr s1, [r2, #32]       @ encoding: [0x08,0x0a,0xd2,0xed]
@ CHECK: vldr s1, [r2, #-32]      @ encoding: [0x08,0x0a,0x52,0xed]
        vldr.32	s1, [r2, #32]
        vldr.32	s1, [r2, #-32]

@ CHECK: vldr s2, [r3]            @ encoding: [0x00,0x1a,0x93,0xed]
        vldr.32 s2, [r3]

@ CHECK: vldr s5, [pc]            @ encoding: [0x00,0x2a,0xdf,0xed]
@ CHECK: vldr s5, [pc]            @ encoding: [0x00,0x2a,0xdf,0xed]
@ CHECK: vldr s5, [pc, #-0]            @ encoding: [0x00,0x2a,0x5f,0xed]
        vldr.32 s5, [pc]
        vldr.32 s5, [pc,#0]
        vldr.32 s5, [pc,#-0]

@ CHECK: vstr d4, [r1]            @ encoding: [0x00,0x4b,0x81,0xed]
@ CHECK: vstr d4, [r1, #24]       @ encoding: [0x06,0x4b,0x81,0xed]
@ CHECK: vstr d4, [r1, #-24]      @ encoding: [0x06,0x4b,0x01,0xed]
@ CHECK: vstr s0, [lr]            @ encoding: [0x00,0x0a,0x8e,0xed]
@ CHECK: vstr d0, [lr]            @ encoding: [0x00,0x0b,0x8e,0xed]

        vstr.64 d4, [r1]
        vstr.64 d4, [r1, #24]
        vstr.64 d4, [r1, #-24]
	vstr s0, [lr]
	vstr d0, [lr]

@ CHECK: vstr s4, [r1]            @ encoding: [0x00,0x2a,0x81,0xed]
@ CHECK: vstr s4, [r1, #24]       @ encoding: [0x06,0x2a,0x81,0xed]
@ CHECK: vstr s4, [r1, #-24]      @ encoding: [0x06,0x2a,0x01,0xed]
        vstr.32 s4, [r1]
        vstr.32 s4, [r1, #24]
        vstr.32 s4, [r1, #-24]

@ CHECK: vldmia r1, {d2, d3, d4, d5, d6, d7} @ encoding: [0x0c,0x2b,0x91,0xec]
@ CHECK: vldmia r1, {s2, s3, s4, s5, s6, s7} @ encoding: [0x06,0x1a,0x91,0xec]
        vldmia  r1, {d2,d3-d6,d7}
        vldmia  r1, {s2,s3-s6,s7}

@ CHECK: vstmia r1, {d2, d3, d4, d5, d6, d7} @ encoding: [0x0c,0x2b,0x81,0xec]
@ CHECK: vstmia	r1, {s2, s3, s4, s5, s6, s7} @ encoding: [0x06,0x1a,0x81,0xec]
@ CHECK: vpush	{d8, d9, d10, d11, d12, d13, d14, d15} @ encoding: [0x10,0x8b,0x2d,0xed]
        vstmia  r1, {d2,d3-d6,d7}
        vstmia  r1, {s2,s3-s6,s7}
        vstmdb sp!, {q4-q7}

@ CHECK: vcvtr.s32.f64  s0, d0 @ encoding: [0x40,0x0b,0xbd,0xee]
@ CHECK: vcvtr.s32.f32  s0, s1 @ encoding: [0x60,0x0a,0xbd,0xee]
@ CHECK: vcvtr.u32.f64  s0, d0 @ encoding: [0x40,0x0b,0xbc,0xee]
@ CHECK: vcvtr.u32.f32  s0, s1 @ encoding: [0x60,0x0a,0xbc,0xee]
        vcvtr.s32.f64  s0, d0
        vcvtr.s32.f32  s0, s1
        vcvtr.u32.f64  s0, d0
        vcvtr.u32.f32  s0, s1

@ CHECK: vmovne	s25, s26, r2, r5
        vmovne	s25, s26, r2, r5        @ encoding: [0x39,0x2a,0x45,0x1c]

@ VMOV w/ optional data type suffix.
	vmov.32 s1, r8
        vmov.s16 s2, r4
        vmov.16 s3, r6
        vmov.u32 s4, r1
        vmov.p8 s5, r2
        vmov.8 s6, r3

        vmov.32 r1, s8
        vmov.s16 r2, s4
        vmov.16 r3, s6
        vmov.u32 r4, s1
        vmov.p8 r5, s2
        vmov.8 r6, s3

@ CHECK: vmov	s1, r8                  @ encoding: [0x90,0x8a,0x00,0xee]
@ CHECK: vmov	s2, r4                  @ encoding: [0x10,0x4a,0x01,0xee]
@ CHECK: vmov	s3, r6                  @ encoding: [0x90,0x6a,0x01,0xee]
@ CHECK: vmov	s4, r1                  @ encoding: [0x10,0x1a,0x02,0xee]
@ CHECK: vmov	s5, r2                  @ encoding: [0x90,0x2a,0x02,0xee]
@ CHECK: vmov	s6, r3                  @ encoding: [0x10,0x3a,0x03,0xee]
@ CHECK: vmov	r1, s8                  @ encoding: [0x10,0x1a,0x14,0xee]
@ CHECK: vmov	r2, s4                  @ encoding: [0x10,0x2a,0x12,0xee]
@ CHECK: vmov	r3, s6                  @ encoding: [0x10,0x3a,0x13,0xee]
@ CHECK: vmov	r4, s1                  @ encoding: [0x90,0x4a,0x10,0xee]
@ CHECK: vmov	r5, s2                  @ encoding: [0x10,0x5a,0x11,0xee]
@ CHECK: vmov	r6, s3                  @ encoding: [0x90,0x6a,0x11,0xee]

# RUN: llvm-mc -triple s390x-linux-gnu < %s | FileCheck %s

#CHECK: .cfi_offset %r0, 0
#CHECK: .cfi_offset %r1, 8
#CHECK: .cfi_offset %r2, 16
#CHECK: .cfi_offset %r3, 24
#CHECK: .cfi_offset %r4, 32
#CHECK: .cfi_offset %r5, 40
#CHECK: .cfi_offset %r6, 48
#CHECK: .cfi_offset %r7, 56
#CHECK: .cfi_offset %r8, 64
#CHECK: .cfi_offset %r9, 72
#CHECK: .cfi_offset %r10, 80
#CHECK: .cfi_offset %r11, 88
#CHECK: .cfi_offset %r12, 96
#CHECK: .cfi_offset %r13, 104
#CHECK: .cfi_offset %r14, 112
#CHECK: .cfi_offset %r15, 120
#CHECK: .cfi_offset %f0, 128
#CHECK: .cfi_offset %f1, 136
#CHECK: .cfi_offset %f2, 144
#CHECK: .cfi_offset %f3, 152
#CHECK: .cfi_offset %f4, 160
#CHECK: .cfi_offset %f5, 168
#CHECK: .cfi_offset %f6, 176
#CHECK: .cfi_offset %f7, 184
#CHECK: .cfi_offset %f8, 192
#CHECK: .cfi_offset %f9, 200
#CHECK: .cfi_offset %f10, 208
#CHECK: .cfi_offset %f11, 216
#CHECK: .cfi_offset %f12, 224
#CHECK: .cfi_offset %f13, 232
#CHECK: .cfi_offset %f14, 240
#CHECK: .cfi_offset %f15, 248

	.cfi_startproc
	.cfi_offset %r0,0
	.cfi_offset %r1,8
	.cfi_offset %r2,16
	.cfi_offset %r3,24
	.cfi_offset %r4,32
	.cfi_offset %r5,40
	.cfi_offset %r6,48
	.cfi_offset %r7,56
	.cfi_offset %r8,64
	.cfi_offset %r9,72
	.cfi_offset %r10,80
	.cfi_offset %r11,88
	.cfi_offset %r12,96
	.cfi_offset %r13,104
	.cfi_offset %r14,112
	.cfi_offset %r15,120
	.cfi_offset %f0,128
	.cfi_offset %f1,136
	.cfi_offset %f2,144
	.cfi_offset %f3,152
	.cfi_offset %f4,160
	.cfi_offset %f5,168
	.cfi_offset %f6,176
	.cfi_offset %f7,184
	.cfi_offset %f8,192
	.cfi_offset %f9,200
	.cfi_offset %f10,208
	.cfi_offset %f11,216
	.cfi_offset %f12,224
	.cfi_offset %f13,232
	.cfi_offset %f14,240
	.cfi_offset %f15,248
	.cfi_endproc

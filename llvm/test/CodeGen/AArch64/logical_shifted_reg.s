	.file	"/home/timnor01/a64-trunk/llvm/test/CodeGen/AArch64/logical_shifted_reg.ll"
	.text
	.globl	logical_32bit
	.type	logical_32bit,@function
logical_32bit:                          // @logical_32bit
	.cfi_startproc
// BB#0:
	adrp	x0, var1_32
	ldr	w1, [x0, #:lo12:var1_32]
	adrp	x0, var2_32
	ldr	w2, [x0, #:lo12:var2_32]
	and	w3, w1, w2
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	bic	w3, w1, w2
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	orr	w3, w1, w2
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	orn	w3, w1, w2
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	eor	w3, w1, w2
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	eon	w3, w2, w1
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	and	w3, w1, w2, lsl #31
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	bic	w3, w1, w2, lsl #31
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	orr	w3, w1, w2, lsl #31
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	orn	w3, w1, w2, lsl #31
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	eor	w3, w1, w2, lsl #31
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	eon	w3, w1, w2, lsl #31
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	bic	w3, w1, w2, asr #10
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	eor	w3, w1, w2, asr #10
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	orn	w3, w1, w2, lsr #1
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	eor	w3, w1, w2, lsr #1
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	eon	w3, w1, w2, ror #20
	adrp	x0, var1_32
	str	w3, [x0, #:lo12:var1_32]
	and	w1, w1, w2, ror #20
	adrp	x0, var1_32
	str	w1, [x0, #:lo12:var1_32]
	ret
.Ltmp0:
	.size	logical_32bit, .Ltmp0-logical_32bit
	.cfi_endproc

	.globl	logical_64bit
	.type	logical_64bit,@function
logical_64bit:                          // @logical_64bit
	.cfi_startproc
// BB#0:
	adrp	x0, var1_64
	ldr	x0, [x0, #:lo12:var1_64]
	adrp	x1, var2_64
	ldr	x1, [x1, #:lo12:var2_64]
	and	x2, x0, x1
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	bic	x2, x0, x1
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	orr	x2, x0, x1
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	orn	x2, x0, x1
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	eor	x2, x0, x1
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	eon	x2, x1, x0
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	and	x2, x0, x1, lsl #63
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	bic	x2, x0, x1, lsl #63
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	orr	x2, x0, x1, lsl #63
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	orn	x2, x0, x1, lsl #63
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	eor	x2, x0, x1, lsl #63
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	eon	x2, x0, x1, lsl #63
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	bic	x2, x0, x1, asr #10
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	eor	x2, x0, x1, asr #10
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	orn	x2, x0, x1, lsr #1
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	eor	x2, x0, x1, lsr #1
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	eon	x2, x0, x1, ror #20
	adrp	x3, var1_64
	str	x2, [x3, #:lo12:var1_64]
	and	x0, x0, x1, ror #20
	adrp	x1, var1_64
	str	x0, [x1, #:lo12:var1_64]
	ret
.Ltmp1:
	.size	logical_64bit, .Ltmp1-logical_64bit
	.cfi_endproc

	.globl	flag_setting
	.type	flag_setting,@function
flag_setting:                           // @flag_setting
	.cfi_startproc
// BB#0:
	sub	sp, sp, #16
	adrp	x0, var1_64
	ldr	x0, [x0, #:lo12:var1_64]
	adrp	x1, var2_64
	ldr	x1, [x1, #:lo12:var2_64]
	tst	x0, x1
	str	x0, [sp, #8]            // 8-byte Folded Spill
	str	x1, [sp]                // 8-byte Folded Spill
	b.gt .LBB2_4
	b	.LBB2_1
.LBB2_1:                                // %test2
	ldr	x0, [sp, #8]            // 8-byte Folded Reload
	ldr	x1, [sp]                // 8-byte Folded Reload
	tst	x0, x1, lsl #63
	b.lt .LBB2_4
	b	.LBB2_2
.LBB2_2:                                // %test3
	ldr	x0, [sp, #8]            // 8-byte Folded Reload
	ldr	x1, [sp]                // 8-byte Folded Reload
	tst	x0, x1, asr #12
	b.gt .LBB2_4
	b	.LBB2_3
.LBB2_3:                                // %other_exit
	adrp	x0, var1_64
	ldr	x1, [sp, #8]            // 8-byte Folded Reload
	str	x1, [x0, #:lo12:var1_64]
	add	sp, sp, #16
	ret
.LBB2_4:                                // %ret
	add	sp, sp, #16
	ret
.Ltmp2:
	.size	flag_setting, .Ltmp2-flag_setting
	.cfi_endproc

	.type	var1_32,@object         // @var1_32
	.bss
	.globl	var1_32
	.align	2
var1_32:
	.word	0                       // 0x0
	.size	var1_32, 4

	.type	var2_32,@object         // @var2_32
	.globl	var2_32
	.align	2
var2_32:
	.word	0                       // 0x0
	.size	var2_32, 4

	.type	var1_64,@object         // @var1_64
	.globl	var1_64
	.align	3
var1_64:
	.xword	0                       // 0x0
	.size	var1_64, 8

	.type	var2_64,@object         // @var2_64
	.globl	var2_64
	.align	3
var2_64:
	.xword	0                       // 0x0
	.size	var2_64, 8



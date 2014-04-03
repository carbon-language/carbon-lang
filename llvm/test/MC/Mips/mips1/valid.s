# Instructions that are valid
#
# FIXME: Test MIPS-I instead of MIPS32
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32 | FileCheck %s

	.set noat
	abs.d	$f7,$f25          # CHECK: encoding:
	abs.s	$f9,$f16
	add	$s7,$s2,$a1
	add.d	$f1,$f7,$f29
	add.s	$f8,$f21,$f24
	addi	$t5,$t1,26322
	addu	$t1,$a0,$a2
	and	$s7,$v0,$t4
	c.ngl.d	$f29,$f29
	c.ngle.d	$f0,$f16
	c.sf.d	$f30,$f0
	c.sf.s	$f14,$f22
	cfc1	$s1,$21
	ctc1	$a2,$26
	cvt.d.s	$f22,$f28
	cvt.d.w	$f26,$f11
	cvt.s.d	$f26,$f8
	cvt.s.w	$f22,$f15
	cvt.w.d	$f20,$f14
	cvt.w.s	$f20,$f24
	div	$zero,$t9,$t3
	div.d	$f29,$f20,$f27
	div.s	$f4,$f5,$f15
	divu	$zero,$t9,$t7
	ehb                      # CHECK: ehb # encoding:  [0x00,0x00,0x00,0xc0]
	lb	$t8,-14515($t2)
	lbu	$t0,30195($v1)
	lh	$t3,-8556($s5)
	lhu	$s3,-22851($v0)
	li	$at,-29773
	li	$zero,-29889
	lw	$t0,5674($a1)
	lwc1	$f16,10225($k0)
	lwc2	$18,-841($a2)
	lwl	$s4,-4231($t7)
	lwr	$zero,-19147($gp)
	mfc1	$a3,$f27
	mfhi	$s3
	mfhi	$sp
	mflo	$s1
	mov.d	$f20,$f14
	mov.s	$f2,$f27
	move	$s8,$a0
	move	$t9,$a2
	mtc1	$s8,$f9
	mthi	$s1
	mtlo	$sp
	mtlo	$t9
	mul.d	$f20,$f20,$f16
	mul.s	$f30,$f10,$f2
	mult	$sp,$s4
	mult	$sp,$v0
	multu	$gp,$k0
	multu	$t1,$s2
	neg.d	$f27,$f18
	neg.s	$f1,$f15
	nop
	nor	$a3,$zero,$a3
	or	$t4,$s0,$sp
	sb	$s6,-19857($t6)
	sh	$t6,-6704($t7)
	sllv	$a3,$zero,$t1
	slt	$s7,$t3,$k1
	slti	$s1,$t2,9489
	sltiu	$t9,$t9,-15531
	sltu	$s4,$s5,$t3
	srav	$s1,$s7,$sp
	srlv	$t9,$s4,$a0
	ssnop                    # CHECK: ssnop # encoding:  [0x00,0x00,0x00,0x40]
	sub	$s6,$s3,$t4
	sub.d	$f18,$f3,$f17
	sub.s	$f23,$f22,$f22
	subu	$sp,$s6,$s6
	sw	$ra,-10160($sp)
	swc1	$f6,-8465($t8)
	swc2	$25,24880($s0)
	swl	$t7,13694($s3)
	swr	$s1,-26590($t6)
	xor	$s2,$a0,$s8

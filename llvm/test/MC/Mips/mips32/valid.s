# Instructions that are valid
#
# RUN: llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32   | FileCheck %s

        .set noat
	abs.d	$f7,$f25 # CHECK: encoding
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
	ceil.w.d	$f11,$f25
	ceil.w.s	$f6,$f20
	cfc1	$s1,$21
	clo	$t3,$a1
	clz	$sp,$gp
	ctc1	$a2,$26
	cvt.d.s	$f22,$f28
	cvt.d.w	$f26,$f11
	cvt.s.d	$f26,$f8
	cvt.s.w	$f22,$f15
	cvt.w.d	$f20,$f14
	cvt.w.s	$f20,$f24
	deret
	div	$zero,$t9,$t3
	div.d	$f29,$f20,$f27
	div.s	$f4,$f5,$f15
	divu	$zero,$t9,$t7
	ehb                      # CHECK: ehb # encoding:  [0x00,0x00,0x00,0xc0]
	eret
	floor.w.d	$f14,$f11
	floor.w.s	$f8,$f9
	lb	$t8,-14515($t2)
	lbu	$t0,30195($v1)
	ldc1	$f11,16391($s0)
	ldc2	$8,-21181($at)
	lh	$t3,-8556($s5)
	lhu	$s3,-22851($v0)
	li	$at,-29773
	li	$zero,-29889
	ll	$v0,-7321($s2)
	lw	$t0,5674($a1)
	lwc1	$f16,10225($k0)
	lwc2	$18,-841($a2)
	lwl	$s4,-4231($t7)
	lwr	$zero,-19147($gp)
	madd	$s6,$t5
	madd	$zero,$t1
	maddu	$s3,$gp
	maddu	$t8,$s2
	mfc0	$a2,$14,1
	mfc1	$a3,$f27
	mfhi	$s3
	mfhi	$sp
	mflo	$s1
	mov.d	$f20,$f14
	mov.s	$f2,$f27
	move	$s8,$a0
	move	$t9,$a2
	movf	$gp,$t0,$fcc7
	movf.d	$f6,$f11,$fcc5
	movf.s	$f23,$f5,$fcc6
	movn	$v1,$s1,$s0
	movn.d	$f27,$f21,$k0
	movn.s	$f12,$f0,$s7
	movt	$zero,$s4,$fcc5
	movt.d	$f0,$f2,$fcc0
	movt.s	$f30,$f2,$fcc1
	movz	$a1,$s6,$t1
	movz.d	$f12,$f29,$t1
	movz.s	$f25,$f7,$v1
	msub	$s7,$k1
	msubu	$t7,$a1
	mtc0	$t1,$29,3
	mtc1	$s8,$f9
	mthi	$s1
	mtlo	$sp
	mtlo	$t9
	mul	$s0,$s4,$at
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
	round.w.d	$f6,$f4
	round.w.s	$f27,$f28
	sb	$s6,-19857($t6)
	sc	$t7,18904($s3)
	sdc1	$f31,30574($t5)
	sdc2	$20,23157($s2)
	sh	$t6,-6704($t7)
	sll   $a3,18               # CHECK: sll $7, $7, 18         # encoding: [0x00,0x07,0x3c,0x80]
	sll   $a3,$zero,18         # CHECK: sll $7, $zero, 18      # encoding: [0x00,0x00,0x3c,0x80]
	sllv  $a3,$t1              # CHECK: sllv $7, $7, $9        # encoding: [0x01,0x27,0x38,0x04]
	sllv  $a3,$zero,$t1        # CHECK: sllv $7, $zero, $9     # encoding: [0x01,0x20,0x38,0x04]
	slt	$s7,$t3,$k1
	slti	$s1,$t2,9489
	sltiu	$t9,$t9,-15531
	sltu	$s4,$s5,$t3
	sqrt.d	$f17,$f22
	sqrt.s	$f0,$f1
	sra   $s1,15               # CHECK: sra $17, $17, 15       # encoding: [0x00,0x11,0x8b,0xc3]
	sra   $s1,$s7,15           # CHECK: sra $17, $23, 15       # encoding: [0x00,0x17,0x8b,0xc3]
	srav  $s1,$sp              # CHECK: srav $17, $17, $sp     # encoding: [0x03,0xb1,0x88,0x07]
	srav  $s1,$s7,$sp          # CHECK: srav $17, $23, $sp     # encoding: [0x03,0xb7,0x88,0x07]
	srl   $2,7                 # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
	srl   $2,$2,7              # CHECK: srl $2, $2, 7          # encoding: [0x00,0x02,0x11,0xc2]
	srlv  $t9,$a0              # CHECK: srlv $25, $25, $4      # encoding: [0x00,0x99,0xc8,0x06]
	srlv  $t9,$s4,$a0          # CHECK: srlv $25, $20, $4      # encoding: [0x00,0x94,0xc8,0x06]
	ssnop                      # CHECK: ssnop                  # encoding: [0x00,0x00,0x00,0x40]
	sub	$s6,$s3,$t4
	sub.d	$f18,$f3,$f17
	sub.s	$f23,$f22,$f22
	subu	$sp,$s6,$s6
	sw	$ra,-10160($sp)
	swc1	$f6,-8465($t8)
	swc2	$25,24880($s0)
	swl	$t7,13694($s3)
	swr	$s1,-26590($t6)
	teqi	$s5,-17504
	tgei	$s1,5025
	tgeiu	$sp,-28621
	tlti	$t6,-21059
	tltiu	$ra,-5076
	tnei	$t4,-29647
	trunc.w.d	$f22,$f15
	trunc.w.s	$f28,$f30
	xor	$s2,$a0,$s8

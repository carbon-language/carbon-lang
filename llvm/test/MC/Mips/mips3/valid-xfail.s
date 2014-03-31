# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# FIXME: Test MIPS-III instead of MIPS64
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64   | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

	.set noat
	ddiv	$zero,$k0,$s3
	ddivu	$zero,$s0,$s1
	div	$zero,$t9,$t3
	divu	$zero,$t9,$t7
	ehb
	lwc3	$10,-32265($k0)
	ssnop
	tlbp
	tlbr
	tlbwi
	tlbwr

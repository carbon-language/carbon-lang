# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips2 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

	.set noat
	ldc3	$29,-28645($s1)
	lwc3	$10,-32265($k0)
	sdc3	$12,5835($t2)
	tlbp
	tlbr
	tlbwi
	tlbwr

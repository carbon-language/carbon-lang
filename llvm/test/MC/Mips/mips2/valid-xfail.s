# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# FIXME: Test MIPS-II instead of MIPS32
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

	.set noat
	ehb
	ldc3	$29,-28645($s1)
	lwc3	$10,-32265($k0)
	sdc3	$12,5835($t2)
	ssnop
	tlbp
	tlbr
	tlbwi
	tlbwr

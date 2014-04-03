# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# FIXME: Test MIPS-I instead of MIPS32
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips32 | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

	.set noat
	tlbp
	tlbr
	tlbwi
	tlbwr
	lwc0	c0_entrylo,-7321($s2)
	lwc3	$10,-32265($k0)
	swc0	c0_prid,18904($s3)

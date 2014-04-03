# Instructions that should be valid but currently fail for known reasons (e.g.
# they aren't implemented yet).
# This test is set up to XPASS if any instruction generates an encoding.
#
# FIXME: Test MIPS-III instead of MIPS64
# RUN: not llvm-mc %s -triple=mips-unknown-linux -show-encoding -mcpu=mips64   | not FileCheck %s
# CHECK-NOT: encoding
# XFAIL: *

	.set noat
	lwc3	$10,-32265($k0)
	tlbp
	tlbr
	tlbwi
	tlbwr

# Instructions that are supposed to be invalid but currently aren't
# This test will XPASS if any insn stops assembling.
#
# RUN: not llvm-mc %s -triple=mips64-unknown-linux -show-encoding -mcpu=mips4 \
# RUN:     2> %t1
# RUN: not FileCheck %s < %t1
# XFAIL: *

# CHECK-NOT: error
        .set noat
	di	$s8
	ei	$t6
	mfhc1	$s8,$f24
	mthc1	$zero,$f16
	rdhwr	$sp,$11

# RUN: llvm-mc -triple mips-unknown-unknown %s

$BB0_2:
	.frame	$sp,0,$ra
	.mask 	0x00000000,0
	.fmask	0x00000000,0
	.set	noreorder
	.set	nomacro
$JTI0_0:
	.gpword	($BB0_2)

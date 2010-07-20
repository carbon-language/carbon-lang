# RUN: llvm-mc -triple i386-pc-linux-gnu %s | FileCheck %s

	.bss
# CHECK: .bss
	.bss


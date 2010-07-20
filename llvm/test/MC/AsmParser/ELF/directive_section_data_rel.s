# RUN: llvm-mc -triple i386-pc-linux-gnu %s | FileCheck %s

	.data.rel
# CHECK: .data.rel
	.data.rel


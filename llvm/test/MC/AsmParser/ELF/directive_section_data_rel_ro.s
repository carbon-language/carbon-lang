# RUN: llvm-mc -triple i386-pc-linux-gnu %s | FileCheck %s

	.data.rel.ro
# CHECK: .data.rel.ro
	.data.rel.ro


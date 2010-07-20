# RUN: llvm-mc -triple i386-pc-linux-gnu %s | FileCheck %s

	.tdata
# CHECK: .tdata
	.tdata


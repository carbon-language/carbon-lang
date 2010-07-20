# RUN: llvm-mc -triple i386-pc-linux-gnu %s | FileCheck %s

	.eh_frame
# CHECK: .eh_frame
	.eh_frame


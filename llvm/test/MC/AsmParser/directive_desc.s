# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .desc foo,16
# CHECK: .desc bar,4
TEST0:  
	.desc foo,0x10
	.desc     bar, 1 +3

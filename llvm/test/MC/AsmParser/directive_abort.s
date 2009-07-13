# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .abort "please stop assembing"
# CHECK: .abort
TEST0:  
	.abort       "please stop assembing"
.abort

# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .abort "please stop assembing"
# CHECK: .abort
TEST0:  
	.abort       "please stop assembing"
.abort

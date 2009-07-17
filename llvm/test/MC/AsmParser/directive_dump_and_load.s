# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .dump "somefile"
# CHECK: .load "jack and jill"
TEST0:  
	.dump       "somefile"
 .load  "jack and jill"

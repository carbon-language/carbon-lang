# RUN: llvm-mc -triple i386-unknown-unknown %s 2> %t
# RUN: FileCheck -input-file %t %s

# CHECK: .abort "please stop assembing"
TEST0:  
	.abort       "please stop assembing"

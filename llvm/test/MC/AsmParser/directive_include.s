# RUN: llvm-mc %s | FileCheck %s

# CHECK: TEST0:
# CHECK: .include "some/include/file"
# CHECK: .include "mary had a little lamb"
TEST0:  
	.include       "some/include/file"
 .include  "mary had a little lamb"

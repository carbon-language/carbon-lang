# RUN: llvm-mc %s -I  %p | FileCheck %s

# CHECK: TESTA:
# CHECK: TEST0:
# CHECK: .set a, 0
# CHECK: TESTB:
TESTA:  
	.include       "directive_set.s"
TESTB:

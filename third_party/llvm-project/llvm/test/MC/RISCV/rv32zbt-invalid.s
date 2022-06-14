# RUN: not llvm-mc -triple riscv32 -mattr=+experimental-zbt < %s 2>&1 | FileCheck %s

# Too few operands
cmix t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
cmov t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
fsl t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
fsr t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Too few operands
fsri t0, t1, t2 # CHECK: :[[@LINE]]:1: error: too few operands for instruction
# Immediate operand out of range
fsri t0, t1, t2, 32 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 31]
fsri t0, t1, t2, -1 # CHECK: :[[@LINE]]:18: error: immediate must be an integer in the range [0, 31]
fslw t0, t1, t2, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
fsrw t0, t1, t2, t3 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set
fsriw t0, t1, t2, 0 # CHECK: :[[@LINE]]:1: error: instruction requires the following: RV64I Base Instruction Set

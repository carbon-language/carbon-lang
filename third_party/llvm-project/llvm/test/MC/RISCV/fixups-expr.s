# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix RELAX %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix RELAX %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix RELAX %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix RELAX %s

# Check that subtraction expressions are emitted as two relocations always.

.globl G1
.globl G2
.L1:
G1:
addi a0, a0, 0
.L2:
G2:

.data
.dword .L2-.L1
.dword G2-G1
.word .L2-.L1
.word G2-G1
.half .L2-.L1
.half G2-G1
.byte .L2-.L1
.byte G2-G1
# RELAX: 0x0 R_RISCV_ADD64 .L2 0x0
# RELAX: 0x0 R_RISCV_SUB64 .L1 0x0
# RELAX: 0x8 R_RISCV_ADD64 G2 0x0
# RELAX: 0x8 R_RISCV_SUB64 G1 0x0
# RELAX: 0x10 R_RISCV_ADD32 .L2 0x0
# RELAX: 0x10 R_RISCV_SUB32 .L1 0x0
# RELAX: 0x14 R_RISCV_ADD32 G2 0x0
# RELAX: 0x14 R_RISCV_SUB32 G1 0x0
# RELAX: 0x18 R_RISCV_ADD16 .L2 0x0
# RELAX: 0x18 R_RISCV_SUB16 .L1 0x0
# RELAX: 0x1A R_RISCV_ADD16 G2 0x0
# RELAX: 0x1A R_RISCV_SUB16 G1 0x0
# RELAX: 0x1C R_RISCV_ADD8 .L2 0x0
# RELAX: 0x1C R_RISCV_SUB8 .L1 0x0
# RELAX: 0x1D R_RISCV_ADD8 G2 0x0
# RELAX: 0x1D R_RISCV_SUB8 G1 0x0

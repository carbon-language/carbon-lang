# RUN: llvm-mc -triple riscv32 -mattr=-relax -riscv-no-aliases < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv32 -mattr=-relax -riscv-no-aliases \
# RUN:     -position-independent < %s | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -position-independent < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv64 -mattr=-relax -riscv-no-aliases < %s \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

# RUN: llvm-mc -triple riscv64 -mattr=-relax -riscv-no-aliases \
# RUN:     -position-independent < %s | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -position-independent < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

.option nopic
# CHECK-INST: .option nopic

la s0, symbol
# CHECK-INST: auipc	s0, %pcrel_hi(symbol)
# CHECK-INST: addi	s0, s0, %pcrel_lo(.Lpcrel_hi0)
# CHECK-RELOC: R_RISCV_PCREL_HI20 symbol 0x0
# CHECK-RELOC: R_RISCV_PCREL_LO12_I .Lpcrel_hi0 0x0


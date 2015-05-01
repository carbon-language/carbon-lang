# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1 --check-prefix=32-BIT
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64 2>&1 | \
# RUN:   FileCheck %s --check-prefix=64-BIT

  .text
  li $5, 0x100000000
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  # 64-BIT: :[[@LINE-2]]:3: error: instruction requires a 32-bit immediate
  dli $5, 1
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 64-bit architecture

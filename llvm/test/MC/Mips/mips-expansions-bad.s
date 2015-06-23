# RUN: not llvm-mc %s -arch=mips -mcpu=mips32r2 2>%t1
# RUN: FileCheck %s < %t1 --check-prefix=32-BIT
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n32 2>&1 | \
# RUN:   FileCheck %s --check-prefix=64-BIT --check-prefix=N32-ONLY
# RUN: not llvm-mc %s -arch=mips64 -mcpu=mips64 -target-abi n64 2>&1 | \
# RUN:   FileCheck %s --check-prefix=64-BIT --check-prefix=N64-ONLY

  .text
  li $5, 0x100000000
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  # 64-BIT: :[[@LINE-2]]:3: error: instruction requires a 32-bit immediate
  la $5, 0x100000000
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  # 64-BIT: :[[@LINE-2]]:3: error: instruction requires a 32-bit immediate
  la $5, 0x100000000($6)
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  # 64-BIT: :[[@LINE-2]]:3: error: instruction requires a 32-bit immediate
  la $5, symbol
  # N64-ONLY: :[[@LINE-1]]:3: warning: instruction loads the 32-bit address of a 64-bit symbol
  # N32-ONLY-NOT: :[[@LINE-2]]:3: warning: instruction loads the 32-bit address of a 64-bit symbol
  dli $5, 1
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 64-bit architecture
  bne $2, 0x100010001, 1332
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  beq $2, 0x100010001, 1332
  # 32-BIT: :[[@LINE-1]]:3: error: instruction requires a 32-bit immediate
  .set mips32r6
  ulhu $5, 0
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  .set mips32
  ulhu $5, 1
  # 32-BIT-NOT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT-NOT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6
  .set mips64r6
  ulhu $5, 2
  # 32-BIT: :[[@LINE-1]]:3: error: instruction not supported on mips32r6 or mips64r6
  # 64-BIT: :[[@LINE-2]]:3: error: instruction not supported on mips32r6 or mips64r6

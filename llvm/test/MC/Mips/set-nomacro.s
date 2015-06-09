# RUN: llvm-mc %s -arch=mips -mcpu=mips32 2>&1 | FileCheck %s

# CHECK-NOT: warning: macro instruction expanded into multiple instructions
  .set macro
  li  $8, -16
  li  $8, 16
  li  $8, 161616

  la  $8, 16
  la  $8, 161616
  la  $8, 16($9)
  la  $8, 161616($9)
  la  $8, symbol

  jal $25
  jal $4, $25

  add $4, $5, $6

  .set noreorder
  .set nomacro
  li  $8, -16
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  li  $8, 16
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  li  $8, 161616
# CHECK: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions

  la  $8, 16
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  la  $8, 161616
# CHECK: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  la  $8, 16($9)
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  la  $8, 161616($9)
# CHECK: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  la  $8, symbol
# CHECK: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions

  jal $25
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions
  jal $4, $25
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions

  add $4, $5, $6
# CHECK-NOT: [[@LINE-1]]:3: warning: macro instruction expanded into multiple instructions

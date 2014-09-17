# RUN: not llvm-mc %s -mcpu=mips32 -mattr=+dsp -triple mips-unknown-linux 2>%t1
# RUN: FileCheck %s < %t1

  lbux    $7, $10($11)

  .set nodsp
  lbux    $6, $10($11)
  # CHECK: error: instruction requires a CPU feature not currently enabled

  .set dsp
  lbux    $5, $10($11)
  # CHECK-NOT: error: instruction requires a CPU feature not currently enabled

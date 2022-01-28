# RUN: llvm-mc %s -triple=mips-unknown-linux -mcpu=mips32r2 -arch=mips 2>&1 -filetype=obj | FileCheck %s
#
# CHECK-NOT: error: out of range PC16 fixup

.text
  b foo
  .space 131072 - 8, 1  # -8 = size of b instr plus size of automatically inserted nop
foo:
  add $0,$0,$0


# REQUIRES: mips
# Check warning on orphaned R_MIPS_HI16 relocations.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe 2>&1 | FileCheck -check-prefix=WARN %s
# RUN: llvm-objdump -d -t --print-imm-hex --no-show-raw-insn %t.exe | FileCheck %s

  .text
  .globl  __start
__start:
  lui    $t0,%hi(__start+0x10000)
  addi   $t0,$t0,%lo(_label)
_label:
  nop

# WARN: can't find matching R_MIPS_LO16 relocation for R_MIPS_HI16

# CHECK: SYMBOL TABLE:
# CHECK: 00020{{0*}}[[VAL:[0-9a-f]+]] l .text   00000000 _label
# CHECK: 00020{{.*}}                  g .text   00000000 __start

# CHECK:      <__start>:
# CHECK-NEXT:  lui    $8, 0x3
#                         ^-- %hi(__start) w/o addend
# CHECK-NEXT:  addi   $8, $8, 0x[[VAL]]
#                             ^-- %lo(_label)

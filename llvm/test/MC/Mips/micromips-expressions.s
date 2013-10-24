# RUN: llvm-mc  %s -triple=mipsel-unknown-linux -show-encoding \
# RUN: -mcpu=mips32r2 -mattr=micromips | FileCheck %s
# Check that the assembler can handle the expressions as operands.
# CHECK:  .text
# CHECK:  .globl  foo
# CHECK:  foo:
# CHECK:  lw   $4, %lo(foo)($4)    # encoding: [0x84'A',0xfc'A',0x00,0x00]
# CHECK:                           # fixup A - offset: 0,
# CHECK:                             value: foo@ABS_LO,
# CHECK:                             kind: fixup_MICROMIPS_LO16
# CHECK:  lw   $4, 56($4)          # encoding: [0x84,0xfc,0x38,0x00]
# CHECK:  lw   $4, %lo(foo+8)($4)  # encoding: [0x84'A',0xfc'A',0x08,0x00]
# CHECK:                           # fixup A - offset: 0,
# CHECK:                             value: foo@ABS_LO,
# CHECK:                             kind: fixup_MICROMIPS_LO16
# CHECK:  lw   $4, %lo(foo+8)($4)  # encoding: [0x84'A',0xfc'A',0x08,0x00]
# CHECK:                           # fixup A - offset: 0,
# CHECK:                             value: foo@ABS_LO,
# CHECK:                             kind: fixup_MICROMIPS_LO16
# CHECK:  lw   $4, %lo(foo+8)($4)  # encoding: [0x84'A',0xfc'A',0x08,0x00]
# CHECK:                           # fixup A - offset: 0,
# CHECK:                             value: foo@ABS_LO,
# CHECK:                             kind: fixup_MICROMIPS_LO16
# CHECK:  .space  64

  .globl  foo
  .ent  foo
foo:
  lw  $4,%lo(foo)($4)
  lw  $4,((10 + 4) * 4)($4)
  lw  $4,%lo (2 * 4) + foo($4)
  lw  $4,%lo((2 * 4) + foo)($4)
  lw  $4,(((%lo ((2 * 4) + foo))))($4)
  .space  64
  .end  foo

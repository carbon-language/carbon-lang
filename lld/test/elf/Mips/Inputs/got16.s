# as -mips32r2 -EL -o got16.o got16.s
    .global glob
    .ent    glob
glob:
    lw      $4,%got(local)($28)
    addiu   $4,$4,%lo(local)
    lw      $4,%got(hidden)($28)
    lw      $4,%call16(glob)($28)
    lw      $4,%call16(extern)($28)
    .end    glob

    .data
    .type   local,%object
    .size   local,4
local:
    .word   undef

    .globl  hidden
    .hidden hidden
    .type   hidden,%object
    .size   hidden,4
hidden:
    .word   0

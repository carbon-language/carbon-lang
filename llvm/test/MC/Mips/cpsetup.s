# RUN: llvm-mc -triple mips64-unknown-unknown -mattr=-n64,+o32 %s | \
# RUN:     FileCheck -check-prefix=ANY -check-prefix=O32 %s
# RUN: llvm-mc -triple mips64-unknown-unknown -mattr=-n64,+n32 %s | \
# RUN:     FileCheck -check-prefix=ANY -check-prefix=NXX -check-prefix=N32 %s
# RUN: llvm-mc -triple mips64-unknown-unknown %s | \
# RUN:     FileCheck -check-prefix=ANY -check-prefix=NXX -check-prefix=N64 %s

# TODO: !PIC -> no output

        .text
        .option pic2
t1:
        .cpsetup $25, 8, __cerror

# ANY-LABEL: t1:

# O32-NOT: __cerror

# NXX: sd       $gp, 8($sp)
# NXX: lui      $gp, %hi(%neg(%gp_rel(__cerror)))
# NXX: addiu    $gp, $gp, %lo(%neg(%gp_rel(__cerror)))
# N32: addu     $gp, $gp, $25
# N64: daddu    $gp, $gp, $25

t2:
# ANY-LABEL: t2:

        .cpsetup $25, $2, __cerror

# O32-NOT: __cerror

# NXX: move     $2, $gp
# NXX: lui      $gp, %hi(%neg(%gp_rel(__cerror)))
# NXX: addiu    $gp, $gp, %lo(%neg(%gp_rel(__cerror)))
# N32: addu     $gp, $gp, $25
# N64: daddu    $gp, $gp, $25

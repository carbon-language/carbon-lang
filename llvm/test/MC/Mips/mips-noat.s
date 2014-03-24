# RUN: not llvm-mc %s -triple=mips-unknown-linux 2>%t0 | FileCheck %s
# RUN: FileCheck -check-prefix=ERROR %s < %t0

# We start with $at enabled
# CHECK-LABEL: test1:
# CHECK:  lui   $1, 1
# CHECK:  addu  $1, $1, $2
# CHECK:  lw    $2, 0($1)
test1:
        lw      $2, 65536($2)

# Check that using $at when .set noat is in effect is an error.
# FIXME: It would be better if the error pointed at the mnemonic instead of the newline
# ERROR: mips-noat.s:[[@LINE+5]]:1: error: Pseudo instruction requires $at, which is not available
# ERROR-NOT: error
test2:
        .set noat
        lw      $2, 65536($2)

# Can we switch it back on successfully?
# CHECK-LABEL: test3:
# CHECK:  lui   $1, 1
# CHECK:  addu  $1, $1, $2
# CHECK:  lw    $2, 0($1)
test3:
        .set at
        lw      $2, 65536($2)

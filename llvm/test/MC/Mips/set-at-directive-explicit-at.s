# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 \
# RUN:     2>%t1 | FileCheck %s
# RUN: FileCheck -check-prefix=WARNINGS %s < %t1
# Check that the assembler can handle the documented syntax
# for ".set at" and set the correct value. The correct value for $at is always
# $1 when written by the user.
    .text
foo:
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
# WARNINGS: :[[@LINE+2]]:11: warning: Used $at without ".set noat"
    .set    at=$1
    jr    $at

# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
# WARNINGS: :[[@LINE+2]]:11: warning: Used $at without ".set noat"
    .set    at=$1
    jr    $1
# WARNINGS-NOT: warning: Used $at without ".set noat"

# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set    at=$2
    jr    $at
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set    at=$3
    jr    $at
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set noat
    jr    $at
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set at=$0
    jr    $at

# CHECK:   jr    $16                     # encoding: [0x08,0x00,0x00,0x02]
# WARNINGS: :[[@LINE+2]]:11: warning: Used $16 with ".set at=$16"
    .set    at=$16
    jr    $s0

# CHECK:   jr    $16                     # encoding: [0x08,0x00,0x00,0x02]
# WARNINGS: :[[@LINE+2]]:11: warning: Used $16 with ".set at=$16"
    .set    at=$16
    jr    $16
# WARNINGS-NOT: warning

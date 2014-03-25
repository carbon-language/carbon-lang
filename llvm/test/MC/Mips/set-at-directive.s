# RUN: llvm-mc %s -triple=mipsel-unknown-linux -show-encoding -mcpu=mips32r2 | \
# RUN: FileCheck %s
# Check that the assembler can handle the documented syntax
# for ".set at" and set the correct value.
# XFAIL:
    .text
foo:
# CHECK:   jr    $1                      # encoding: [0x08,0x00,0x20,0x00]
    .set    at=$1
    jr    $at
    nop
# CHECK:   jr    $2                      # encoding: [0x08,0x00,0x40,0x00]
    .set    at=$2
    jr    $at
    nop
# CHECK:   jr    $3                      # encoding: [0x08,0x00,0x60,0x00]
    .set    at=$3
    jr    $at
    nop
# CHECK:   jr    $4                      # encoding: [0x08,0x00,0x80,0x00]
    .set    at=$a0
    jr    $at
    nop
# CHECK:   jr    $5                      # encoding: [0x08,0x00,0xa0,0x00]
    .set    at=$a1
    jr    $at
    nop
# CHECK:   jr    $6                      # encoding: [0x08,0x00,0xc0,0x00]
    .set    at=$a2
    jr    $at
    nop
# CHECK:   jr $7                # encoding: [0x08,0x00,0xe0,0x00]
    .set    at=$a3
    jr    $at
    nop
# CHECK:   jr    $8                      # encoding: [0x08,0x00,0x00,0x01]
    .set    at=$8
    jr    $at
    nop
# CHECK:   jr    $9                      # encoding: [0x08,0x00,0x20,0x01]
    .set    at=$9
    jr    $at
    nop
# CHECK:   jr    $10                     # encoding: [0x08,0x00,0x40,0x01]
    .set    at=$10
    jr    $at
    nop
# CHECK:   jr    $11                     # encoding: [0x08,0x00,0x60,0x01]
    .set    at=$11
    jr    $at
    nop
# CHECK:   jr    $12                     # encoding: [0x08,0x00,0x80,0x01]
    .set    at=$12
    jr    $at
    nop
# CHECK:   jr    $13                     # encoding: [0x08,0x00,0xa0,0x01]
    .set    at=$13
    jr    $at
    nop
# CHECK:   jr    $14                     # encoding: [0x08,0x00,0xc0,0x01]
    .set    at=$14
    jr    $at
    nop
# CHECK:   jr    $15                     # encoding: [0x08,0x00,0xe0,0x01]
    .set    at=$15
    jr    $at
    nop
# CHECK:   jr    $16                     # encoding: [0x08,0x00,0x00,0x02]
    .set    at=$s0
    jr    $at
    nop
# CHECK:   jr    $17                     # encoding: [0x08,0x00,0x20,0x02]
    .set    at=$s1
    jr    $at
    nop
# CHECK:   jr    $18                     # encoding: [0x08,0x00,0x40,0x02]
    .set    at=$s2
    jr    $at
    nop
# CHECK:   jr    $19                     # encoding: [0x08,0x00,0x60,0x02]
    .set    at=$s3
    jr    $at
    nop
# CHECK:   jr    $20                     # encoding: [0x08,0x00,0x80,0x02]
    .set    at=$s4
    jr    $at
    nop
# CHECK:   jr    $21                     # encoding: [0x08,0x00,0xa0,0x02]
    .set    at=$s5
    jr    $at
    nop
# CHECK:   jr    $22                     # encoding: [0x08,0x00,0xc0,0x02]
    .set    at=$s6
    jr    $at
    nop
# CHECK:   jr    $23                     # encoding: [0x08,0x00,0xe0,0x02]
    .set    at=$s7
    jr    $at
    nop
# CHECK:   jr    $24                     # encoding: [0x08,0x00,0x00,0x03]
    .set    at=$24
    jr    $at
    nop
# CHECK:   jr    $25                     # encoding: [0x08,0x00,0x20,0x03]
    .set    at=$25
    jr    $at
    nop
# CHECK:   jr    $26                     # encoding: [0x08,0x00,0x40,0x03]
    .set    at=$26
    jr    $at
    nop
# CHECK:   jr    $27                     # encoding: [0x08,0x00,0x60,0x03]
    .set    at=$27
    jr    $at
    nop
# CHECK:   jr    $gp                     # encoding: [0x08,0x00,0x80,0x03]
    .set    at=$gp
    jr    $at
    nop
# CHECK:   jr    $fp                     # encoding: [0x08,0x00,0xc0,0x03]
    .set    at=$fp
    jr    $at
    nop
# CHECK:   jr    $sp                     # encoding: [0x08,0x00,0xa0,0x03]
    .set    at=$sp
    jr    $at
    nop
# CHECK:   jr    $ra                     # encoding: [0x08,0x00,0xe0,0x03]
    .set    at=$ra
    jr    $at
    nop

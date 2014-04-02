# RUN: llvm-mc  %s -triple=mips64-unknown-linux -show-encoding -mcpu=octeon | FileCheck %s

# CHECK: baddu $9, $6, $7             # encoding: [0x70,0xc7,0x48,0x28]
# CHECK: baddu $17, $18, $19          # encoding: [0x72,0x53,0x88,0x28]
# CHECK: baddu $2, $2, $3             # encoding: [0x70,0x43,0x10,0x28]
# CHECK: cins  $25, $10, 22, 2        # encoding: [0x71,0x59,0x15,0xb2]
# CHECK: cins  $9, $9, 17, 29         # encoding: [0x71,0x29,0xec,0x72]
# CHECK: cins32 $15, $2, 18, 8        # encoding: [0x70,0x4f,0x44,0xb3]
# CHECK: cins32 $22, $22, 9, 22       # encoding: [0x72,0xd6,0xb2,0x73]
# CHECK: dmul  $9, $6, $7             # encoding: [0x70,0xc7,0x48,0x03]
# CHECK: dmul  $19, $24, $25          # encoding: [0x73,0x19,0x98,0x03]
# CHECK: dmul  $9, $9, $6             # encoding: [0x71,0x26,0x48,0x03]
# CHECK: dmul  $21, $21, $25          # encoding: [0x72,0xb9,0xa8,0x03]
# CHECK: dpop  $9, $6                 # encoding: [0x70,0xc0,0x48,0x2d]
# CHECK: dpop  $15, $22               # encoding: [0x72,0xc0,0x78,0x2d]
# CHECK: dpop  $12, $12               # encoding: [0x71,0x80,0x60,0x2d]
# CHECK: exts  $4, $25, 27, 15        # encoding: [0x73,0x24,0x7e,0xfa]
# CHECK: exts  $15, $15, 17, 6        # encoding: [0x71,0xef,0x34,0x7a]
# CHECK: exts32 $4, $13, 10, 8        # encoding: [0x71,0xa4,0x42,0xbb]
# CHECK: exts32 $15, $15, 11, 20      # encoding: [0x71,0xef,0xa2,0xfb]
# CHECK: mtm0  $15                    # encoding: [0x71,0xe0,0x00,0x08]
# CHECK: mtm1  $16                    # encoding: [0x72,0x00,0x00,0x0c]
# CHECK: mtm2  $17                    # encoding: [0x72,0x20,0x00,0x0d]
# CHECK: mtp0  $18                    # encoding: [0x72,0x40,0x00,0x09]
# CHECK: mtp1  $19                    # encoding: [0x72,0x60,0x00,0x0a]
# CHECK: mtp2  $20                    # encoding: [0x72,0x80,0x00,0x0b]
# CHECK: pop   $9, $6                 # encoding: [0x70,0xc0,0x48,0x2c]
# CHECK: pop   $8, $19                # encoding: [0x72,0x60,0x40,0x2c]
# CHECK: pop   $2, $2                 # encoding: [0x70,0x40,0x10,0x2c]
# CHECK: seq   $25, $23, $24          # encoding: [0x72,0xf8,0xc8,0x2a]
# CHECK: seq   $6, $6, $24            # encoding: [0x70,0xd8,0x30,0x2a]
# CHECK: sne   $25, $23, $24          # encoding: [0x72,0xf8,0xc8,0x2b]
# CHECK: sne   $23, $23, $20          # encoding: [0x72,0xf4,0xb8,0x2b]

  baddu $9, $6, $7
  baddu $17, $18, $19
  baddu $2, $3
  cins  $25, $10, 22, 2
  cins  $9, 17, 29
  cins32 $15, $2, 18, 8
  cins32 $22, 9, 22
  dmul  $9, $6, $7
  dmul  $19, $24, $25
  dmul  $9, $6
  dmul  $21, $25
  dpop  $9, $6
  dpop  $15, $22
  dpop  $12
  exts  $4, $25, 27, 15
  exts  $15, 17, 6
  exts32 $4, $13, 10, 8
  exts32 $15, 11, 20
  mtm0  $15
  mtm1  $16
  mtm2  $17
  mtp0  $18
  mtp1  $19
  mtp2  $20
  pop   $9, $6
  pop   $8, $19
  pop   $2
  seq   $25, $23, $24
  seq   $6, $24
  sne   $25, $23, $24
  sne   $23, $20

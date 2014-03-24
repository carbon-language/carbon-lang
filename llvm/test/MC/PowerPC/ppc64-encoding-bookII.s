
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Cache management instruction

# CHECK-BE: icbi 2, 3                       # encoding: [0x7c,0x02,0x1f,0xac]
# CHECK-LE: icbi 2, 3                       # encoding: [0xac,0x1f,0x02,0x7c]
            icbi 2, 3

# FIXME:    dcbt 2, 3, 10
# CHECK-BE: dcbt 2, 3                       # encoding: [0x7c,0x02,0x1a,0x2c]
# CHECK-LE: dcbt 2, 3                       # encoding: [0x2c,0x1a,0x02,0x7c]
            dcbt 2, 3
# FIXME:    dcbtst 2, 3, 10
# CHECK-BE: dcbtst 2, 3                     # encoding: [0x7c,0x02,0x19,0xec]
# CHECK-LE: dcbtst 2, 3                     # encoding: [0xec,0x19,0x02,0x7c]
            dcbtst 2, 3
# CHECK-BE: dcbz 2, 3                       # encoding: [0x7c,0x02,0x1f,0xec]
# CHECK-LE: dcbz 2, 3                       # encoding: [0xec,0x1f,0x02,0x7c]
            dcbz 2, 3
# CHECK-BE: dcbst 2, 3                      # encoding: [0x7c,0x02,0x18,0x6c]
# CHECK-LE: dcbst 2, 3                      # encoding: [0x6c,0x18,0x02,0x7c]
            dcbst 2, 3
# FIXME:    dcbf 2, 3, 1

# Synchronization instructions

# CHECK-BE: isync                           # encoding: [0x4c,0x00,0x01,0x2c]
# CHECK-LE: isync                           # encoding: [0x2c,0x01,0x00,0x4c]
            isync

# FIXME:    lbarx 2, 3, 4, 1
# FIXME:    lharx 2, 3, 4, 1
# FIXME:    lwarx 2, 3, 4, 1
# FIXME:    ldarx 2, 3, 4, 1

# FIXME:    stbcx. 2, 3, 4
# FIXME:    sthcx. 2, 3, 4
# CHECK-BE: stwcx. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x2d]
# CHECK-LE: stwcx. 2, 3, 4                  # encoding: [0x2d,0x21,0x43,0x7c]
            stwcx. 2, 3, 4
# CHECK-BE: stdcx. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0xad]
# CHECK-LE: stdcx. 2, 3, 4                  # encoding: [0xad,0x21,0x43,0x7c]
            stdcx. 2, 3, 4

# CHECK-BE: sync 2                          # encoding: [0x7c,0x40,0x04,0xac]
# CHECK-LE: sync 2                          # encoding: [0xac,0x04,0x40,0x7c]
            sync 2
# CHECK-BE: eieio                           # encoding: [0x7c,0x00,0x06,0xac]
# CHECK-LE: eieio                           # encoding: [0xac,0x06,0x00,0x7c]
            eieio
# CHECK-BE: wait 2                          # encoding: [0x7c,0x40,0x00,0x7c]
# CHECK-LE: wait 2                          # encoding: [0x7c,0x00,0x40,0x7c]
            wait 2

# Extended mnemonics

# CHECK-BE: dcbf 2, 3                       # encoding: [0x7c,0x02,0x18,0xac]
# CHECK-LE: dcbf 2, 3                       # encoding: [0xac,0x18,0x02,0x7c]
            dcbf 2, 3
# FIXME:    dcbfl 2, 3

# FIXME:    lbarx 2, 3, 4
# FIXME:    lharx 2, 3, 4
# CHECK-BE: lwarx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x28]
# CHECK-LE: lwarx 2, 3, 4                   # encoding: [0x28,0x20,0x43,0x7c]
            lwarx 2, 3, 4
# CHECK-BE: ldarx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0xa8]
# CHECK-LE: ldarx 2, 3, 4                   # encoding: [0xa8,0x20,0x43,0x7c]
            ldarx 2, 3, 4

# CHECK-BE: sync 0                          # encoding: [0x7c,0x00,0x04,0xac]
# CHECK-LE: sync 0                          # encoding: [0xac,0x04,0x00,0x7c]
            sync
# CHECK-BE: sync 0                          # encoding: [0x7c,0x00,0x04,0xac]
# CHECK-LE: sync 0                          # encoding: [0xac,0x04,0x00,0x7c]
            msync
# CHECK-BE: sync 1                          # encoding: [0x7c,0x20,0x04,0xac]
# CHECK-LE: sync 1                          # encoding: [0xac,0x04,0x20,0x7c]
            lwsync
# CHECK-BE: sync 2                          # encoding: [0x7c,0x40,0x04,0xac]
# CHECK-LE: sync 2                          # encoding: [0xac,0x04,0x40,0x7c]
            ptesync

# CHECK-BE: wait 0                          # encoding: [0x7c,0x00,0x00,0x7c]
# CHECK-LE: wait 0                          # encoding: [0x7c,0x00,0x00,0x7c]
            wait
# CHECK-BE: wait 1                          # encoding: [0x7c,0x20,0x00,0x7c]
# CHECK-LE: wait 1                          # encoding: [0x7c,0x00,0x20,0x7c]
            waitrsv
# CHECK-BE: wait 2                          # encoding: [0x7c,0x40,0x00,0x7c]
# CHECK-LE: wait 2                          # encoding: [0x7c,0x00,0x40,0x7c]
            waitimpl

# Time base instructions

# CHECK-BE: mftb 2, 123                     # encoding: [0x7c,0x5b,0x1a,0xe6]
# CHECK-LE: mftb 2, 123                     # encoding: [0xe6,0x1a,0x5b,0x7c]
            mftb 2, 123
# CHECK-BE: mftb 2, 268                     # encoding: [0x7c,0x4c,0x42,0xe6]
# CHECK-LE: mftb 2, 268                     # encoding: [0xe6,0x42,0x4c,0x7c]
            mftb 2
# CHECK-BE: mftb 2, 269                     # encoding: [0x7c,0x4d,0x42,0xe6]
# CHECK-LE: mftb 2, 269                     # encoding: [0xe6,0x42,0x4d,0x7c]
            mftbu 2


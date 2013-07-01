
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# Cache management instruction

# FIXME: icbi 2, 3
# FIXME: icbt 1, 2, 3

# FIXME: dcbt 2, 3, 10
# FIXME: dcbtst 2, 3, 10
# CHECK: dcbz 2, 3                       # encoding: [0x7c,0x02,0x1f,0xec]
         dcbz 2, 3
# CHECK: dcbst 2, 3                      # encoding: [0x7c,0x02,0x18,0x6c]
         dcbst 2, 3
# FIXME: dcbf 2, 3, 1

# Synchronization instructions

# FIXME: isync

# FIXME: lbarx 2, 3, 4, 1
# FIXME: lharx 2, 3, 4, 1
# FIXME: lwarx 2, 3, 4, 1
# FIXME: ldarx 2, 3, 4, 1

# FIXME: stbcx. 2, 3, 4
# FIXME: sthcx. 2, 3, 4
# CHECK: stwcx. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0x2d]
         stwcx. 2, 3, 4
# CHECK: stdcx. 2, 3, 4                  # encoding: [0x7c,0x43,0x21,0xad]
         stdcx. 2, 3, 4

# CHECK: sync 2                          # encoding: [0x7c,0x40,0x04,0xac]
         sync 2
# FIXME: eieio
# FIXME: wait 2

# Extended mnemonics

# CHECK: dcbf 2, 3                       # encoding: [0x7c,0x02,0x18,0xac]
         dcbf 2, 3
# FIXME: dcbfl 2, 3

# FIXME: lbarx 2, 3, 4
# FIXME: lharx 2, 3, 4
# CHECK: lwarx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0x28]
         lwarx 2, 3, 4
# CHECK: ldarx 2, 3, 4                   # encoding: [0x7c,0x43,0x20,0xa8]
         ldarx 2, 3, 4

# CHECK: sync 0                          # encoding: [0x7c,0x00,0x04,0xac]
         sync
# CHECK: sync 1                          # encoding: [0x7c,0x20,0x04,0xac]
         lwsync
# CHECK: sync 2                          # encoding: [0x7c,0x40,0x04,0xac]
         ptesync

# FIXME: wait
# FIXME: waitrsv
# FIXME: waitimpl


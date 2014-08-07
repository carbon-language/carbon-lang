# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-BE %s
# RUN: llvm-mc -triple powerpc64le-unknown-unknown --show-encoding %s | FileCheck -check-prefix=CHECK-LE %s

# Instructions from the Signal Processing Engine extension:

# CHECK-BE: evlddx 14, 21, 28               # encoding: [0x11,0xd5,0xe3,0x00]
# CHECK-LE: evlddx 14, 21, 28               # encoding: [0x00,0xe3,0xd5,0x11]
evlddx %r14, %r21, %r28
# CHECK-BE: evldwx 14, 21, 28               # encoding: [0x11,0xd5,0xe3,0x02]
# CHECK-LE: evldwx 14, 21, 28               # encoding: [0x02,0xe3,0xd5,0x11]
evldwx %r14, %r21, %r28
# CHECK-BE: evldhx 14, 21, 28               # encoding: [0x11,0xd5,0xe3,0x04]
# CHECK-LE: evldhx 14, 21, 28               # encoding: [0x04,0xe3,0xd5,0x11]
evldhx %r14, %r21, %r28
# CHECK-BE: evlhhesplatx 14, 21, 28         # encoding: [0x11,0xd5,0xe3,0x08]
# CHECK-LE: evlhhesplatx 14, 21, 28         # encoding: [0x08,0xe3,0xd5,0x11]
evlhhesplatx %r14, %r21, %r28
# CHECK-BE: evlhhousplatx 14, 21, 28        # encoding: [0x11,0xd5,0xe3,0x0c]
# CHECK-LE: evlhhousplatx 14, 21, 28        # encoding: [0x0c,0xe3,0xd5,0x11]
evlhhousplatx %r14, %r21, %r28
# CHECK-BE: evlhhossplatx 14, 21, 28        # encoding: [0x11,0xd5,0xe3,0x0e]
# CHECK-LE: evlhhossplatx 14, 21, 28        # encoding: [0x0e,0xe3,0xd5,0x11]
evlhhossplatx %r14, %r21, %r28
# CHECK-BE: evlwhex 14, 21, 28              # encoding: [0x11,0xd5,0xe3,0x10]
# CHECK-LE: evlwhex 14, 21, 28              # encoding: [0x10,0xe3,0xd5,0x11]
evlwhex %r14, %r21, %r28
# CHECK-BE: evlwhoux 14, 21, 28             # encoding: [0x11,0xd5,0xe3,0x14]
# CHECK-LE: evlwhoux 14, 21, 28             # encoding: [0x14,0xe3,0xd5,0x11]
evlwhoux %r14, %r21, %r28
# CHECK-BE: evlwhosx 14, 21, 28             # encoding: [0x11,0xd5,0xe3,0x16]
# CHECK-LE: evlwhosx 14, 21, 28             # encoding: [0x16,0xe3,0xd5,0x11]
evlwhosx %r14, %r21, %r28
# CHECK-BE: evlwwsplatx 14, 21, 28          # encoding: [0x11,0xd5,0xe3,0x18]
# CHECK-LE: evlwwsplatx 14, 21, 28          # encoding: [0x18,0xe3,0xd5,0x11]
evlwwsplatx %r14, %r21, %r28
# CHECK-BE: evlwhsplatx 14, 21, 28          # encoding: [0x11,0xd5,0xe3,0x1c]
# CHECK-LE: evlwhsplatx 14, 21, 28          # encoding: [0x1c,0xe3,0xd5,0x11]
evlwhsplatx %r14, %r21, %r28
# CHECK-BE: evmergehi 14, 21, 28            # encoding: [0x11,0xd5,0xe2,0x2c]
# CHECK-LE: evmergehi 14, 21, 28            # encoding: [0x2c,0xe2,0xd5,0x11]
evmergehi %r14, %r21, %r28
# CHECK-BE: evmergelo 14, 21, 28            # encoding: [0x11,0xd5,0xe2,0x2d]
# CHECK-LE: evmergelo 14, 21, 28            # encoding: [0x2d,0xe2,0xd5,0x11]
evmergelo %r14, %r21, %r28
# CHECK-BE: evmergehilo 14, 21, 28          # encoding: [0x11,0xd5,0xe2,0x2e]
# CHECK-LE: evmergehilo 14, 21, 28          # encoding: [0x2e,0xe2,0xd5,0x11]
evmergehilo %r14, %r21, %r28
# CHECK-BE: evmergelohi 14, 21, 28          # encoding: [0x11,0xd5,0xe2,0x2f]
# CHECK-LE: evmergelohi 14, 21, 28          # encoding: [0x2f,0xe2,0xd5,0x11]
evmergelohi %r14, %r21, %r28

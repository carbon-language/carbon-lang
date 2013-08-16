# RUN: llvm-mc %s -triple=mips-unknown-unknown -show-encoding -mcpu=mips32r2 | \
# RUN: FileCheck -check-prefix=CHECK32  %s
# RUN: llvm-mc %s -triple=mips-unknown-unknown -show-encoding -mcpu=mips64r2 | \
# RUN: FileCheck -check-prefix=CHECK64  %s

# CHECK32:    break                      # encoding: [0x00,0x00,0x00,0x0d]
# CHECK32:    break   7, 0               # encoding: [0x00,0x07,0x00,0x0d]
# CHECK32:    break   7, 5               # encoding: [0x00,0x07,0x01,0x4d]
# CHECK32:    syscall                    # encoding: [0x00,0x00,0x00,0x0c]
# CHECK32:    syscall 13396              # encoding: [0x00,0x0d,0x15,0x0c]
# CHECK32:    eret                       # encoding: [0x42,0x00,0x00,0x18]
# CHECK32:    deret                      # encoding: [0x42,0x00,0x00,0x1f]
# CHECK32:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK32:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK32:    di      $10                # encoding: [0x41,0x6a,0x60,0x00]
# CHECK32:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK32:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK32:    ei      $10                # encoding: [0x41,0x6a,0x60,0x20]
# CHECK32:    wait                       # encoding: [0x42,0x00,0x00,0x20]

# CHECK64:    break                      # encoding: [0x00,0x00,0x00,0x0d]
# CHECK64:    break   7, 0               # encoding: [0x00,0x07,0x00,0x0d]
# CHECK64:    break   7, 5               # encoding: [0x00,0x07,0x01,0x4d]
# CHECK64:    syscall                    # encoding: [0x00,0x00,0x00,0x0c]
# CHECK64:    syscall 13396              # encoding: [0x00,0x0d,0x15,0x0c]
# CHECK64:    eret                       # encoding: [0x42,0x00,0x00,0x18]
# CHECK64:    deret                      # encoding: [0x42,0x00,0x00,0x1f]
# CHECK64:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK64:    di                         # encoding: [0x41,0x60,0x60,0x00]
# CHECK64:    di      $10                # encoding: [0x41,0x6a,0x60,0x00]
# CHECK64:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK64:    ei                         # encoding: [0x41,0x60,0x60,0x20]
# CHECK64:    ei      $10                # encoding: [0x41,0x6a,0x60,0x20]
# CHECK64:    wait                       # encoding: [0x42,0x00,0x00,0x20]
    break
    break 7
    break 7,5
    syscall
    syscall 0x3454
    eret
    deret
    di
    di  $0
    di  $10

    ei
    ei  $0
    ei  $10

    wait

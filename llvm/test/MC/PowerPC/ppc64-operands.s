
# RUN: llvm-mc -triple powerpc64-unknown-unknown --show-encoding %s | FileCheck %s

# Register operands

# CHECK: add 1, 2, 3                     # encoding: [0x7c,0x22,0x1a,0x14]
         add 1, 2, 3

# CHECK: add 1, 2, 3                     # encoding: [0x7c,0x22,0x1a,0x14]
         add %r1, %r2, %r3

# CHECK: add 0, 0, 0                     # encoding: [0x7c,0x00,0x02,0x14]
         add 0, 0, 0

# CHECK: add 31, 31, 31                  # encoding: [0x7f,0xff,0xfa,0x14]
         add 31, 31, 31

# CHECK: addi 1, 0, 0                    # encoding: [0x38,0x20,0x00,0x00]
         addi 1, 0, 0

# CHECK: addi 1, 0, 0                    # encoding: [0x38,0x20,0x00,0x00]
         addi 1, %r0, 0

# Signed 16-bit immediate operands

# CHECK: addi 1, 2, 0                    # encoding: [0x38,0x22,0x00,0x00]
         addi 1, 2, 0

# CHECK: addi 1, 0, -32768               # encoding: [0x38,0x20,0x80,0x00]
         addi 1, 0, -32768

# CHECK: addi 1, 0, 32767                # encoding: [0x38,0x20,0x7f,0xff]
         addi 1, 0, 32767

# Unsigned 16-bit immediate operands

# CHECK: ori 1, 2, 0                     # encoding: [0x60,0x41,0x00,0x00]
         ori 1, 2, 0

# CHECK: ori 1, 2, 65535                 # encoding: [0x60,0x41,0xff,0xff]
         ori 1, 2, 65535

# Signed 16-bit immediate operands (extended range for addis)

# CHECK: addis 1, 0, 0                   # encoding: [0x3c,0x20,0x00,0x00]
         addis 1, 0, -65536

# CHECK: addis 1, 0, -1                  # encoding: [0x3c,0x20,0xff,0xff]
         addis 1, 0, 65535

# D-Form memory operands

# CHECK: lwz 1, 0(0)                     # encoding: [0x80,0x20,0x00,0x00]
         lwz 1, 0(0)

# CHECK: lwz 1, 0(0)                     # encoding: [0x80,0x20,0x00,0x00]
         lwz 1, 0(%r0)

# CHECK: lwz 1, 0(31)                    # encoding: [0x80,0x3f,0x00,0x00]
         lwz 1, 0(31)

# CHECK: lwz 1, 0(31)                    # encoding: [0x80,0x3f,0x00,0x00]
         lwz 1, 0(%r31)

# CHECK: lwz 1, -32768(2)                # encoding: [0x80,0x22,0x80,0x00]
         lwz 1, -32768(2)

# CHECK: lwz 1, 32767(2)                 # encoding: [0x80,0x22,0x7f,0xff]
         lwz 1, 32767(2)


# CHECK: ld 1, 0(0)                      # encoding: [0xe8,0x20,0x00,0x00]
         ld 1, 0(0)

# CHECK: ld 1, 0(0)                      # encoding: [0xe8,0x20,0x00,0x00]
         ld 1, 0(%r0)

# CHECK: ld 1, 0(31)                     # encoding: [0xe8,0x3f,0x00,0x00]
         ld 1, 0(31)

# CHECK: ld 1, 0(31)                     # encoding: [0xe8,0x3f,0x00,0x00]
         ld 1, 0(%r31)

# CHECK: ld 1, -32768(2)                 # encoding: [0xe8,0x22,0x80,0x00]
         ld 1, -32768(2)

# CHECK: ld 1, 32764(2)                  # encoding: [0xe8,0x22,0x7f,0xfc]
         ld 1, 32764(2)

# CHECK: ld 1, 4(2)                      # encoding: [0xe8,0x22,0x00,0x04]
         ld 1, 4(2)

# CHECK: ld 1, -4(2)                     # encoding: [0xe8,0x22,0xff,0xfc]
         ld 1, -4(2)


# Immediate branch operands

# CHECK: b .+1024                        # encoding: [0x48,0x00,0x04,0x00]
         b 1024

# CHECK: ba 1024                         # encoding: [0x48,0x00,0x04,0x02]
         ba 1024

# CHECK: beq 0, .+1024                   # encoding: [0x41,0x82,0x04,0x00]
         beq 1024

# CHECK: beqa 0, 1024                    # encoding: [0x41,0x82,0x04,0x02]
         beqa 1024


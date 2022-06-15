## Test valid misc instructions.

# RUN: llvm-mc %s --triple=loongarch32 --show-encoding \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --show-encoding --defsym=LA64=1 \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ,CHECK64-ASM,CHECK64-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch32 --filetype=obj | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s --triple=loongarch64 --filetype=obj --defsym=LA64=1 | llvm-objdump -d - \
# RUN:     | FileCheck --check-prefixes=CHECK-ASM-AND-OBJ,CHECK64-ASM-AND-OBJ %s

#############################################################
## Instructions for both loongarch32 and loongarch64
#############################################################

# CHECK-ASM-AND-OBJ: syscall 100
# CHECK-ASM: encoding: [0x64,0x00,0x2b,0x00]
syscall 100

# CHECK-ASM-AND-OBJ: break 199
# CHECK-ASM: encoding: [0xc7,0x00,0x2a,0x00]
break 199

# CHECK-ASM-AND-OBJ: rdtimel.w $s1, $a0
# CHECK-ASM: encoding: [0x98,0x60,0x00,0x00]
rdtimel.w $s1, $a0

# CHECK-ASM-AND-OBJ: rdtimeh.w $a7, $a1
# CHECK-ASM: encoding: [0xab,0x64,0x00,0x00]
rdtimeh.w $a7, $a1

# CHECK-ASM-AND-OBJ: cpucfg $sp, $a4
# CHECK-ASM: encoding: [0x03,0x6d,0x00,0x00]
cpucfg $sp, $a4


#############################################################
## Instructions only for loongarch64
#############################################################

.ifdef LA64

# CHECK64-ASM-AND-OBJ: asrtle.d $t0, $t5
# CHECK64-ASM: encoding: [0x80,0x45,0x01,0x00]
asrtle.d $t0, $t5

# CHECK64-ASM-AND-OBJ: asrtgt.d $t8, $t8
# CHECK64-ASM: encoding: [0x80,0xd2,0x01,0x00]
asrtgt.d $t8, $t8

# CHECK64-ASM-AND-OBJ: rdtime.d $tp, $t3
# CHECK64-ASM: encoding: [0xe2,0x69,0x00,0x00]
rdtime.d $tp, $t3

.endif


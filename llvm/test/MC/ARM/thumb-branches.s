@ RUN: llvm-mc < %s -triple thumbv5-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -r | FileCheck %s


        bl      end
        .space 0x3fffff
end:

        bl      end2
        .space 0x3fffff
        .global end2
end2:

        bl      end3
        .space 0x400000
        .global end3
end3:

        bl      end4
        .space 0x400000
end4:

@ CHECK: 0x400003 R_ARM_THM_CALL end2 0x0
@ CHECK: 0x800006 R_ARM_THM_CALL end3 0x0
@ CHECK: 0xC0000A R_ARM_THM_CALL end4 0x0

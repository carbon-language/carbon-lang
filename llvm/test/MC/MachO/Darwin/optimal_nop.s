// Validate that we can assemble this file exactly like the platform
// assembler.
//
// RUN: llvm-mc -filetype=obj -triple i386-apple-darwin10 -o %t.mc.o %s
// RUN: as -arch i386 -o %t.as.o %s
// RUN: diff %t.mc.o %t.as.o

# 1 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nop
        # 0x90
        .align 1, 0x90
        ret
# 2 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # xchg %ax,%ax
        # 0x66, 0x90
        .align 2, 0x90
        ret
# 3 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl (%[re]ax)
        # 0x0f, 0x1f, 0x00
        .align 2, 0x90
        ret
# 4 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopl 0(%[re]ax)
        # 0x0f, 0x1f, 0x40, 0x00
        .align 3, 0x90
        ret
# 5 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopl 0(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 6 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 3, 0x90
        ret
# 7 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 8 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 3, 0x90
        ret
# 9 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw 0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 10 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 11 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        ret
        # nopw %cs:0L(%[re]ax,%[re]ax,1)
        # 0x66, 0x2e, 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 12 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopw 0(%[re]ax,%[re]ax,1)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00
        .align 4, 0x90
        ret
# 13 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        ret
        # nopw 0(%[re]ax,%[re]ax,1)
        # nopl 0L(%[re]ax)
        # 0x66, 0x0f, 0x1f, 0x44, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 14 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret
# 15 byte nop test
        .align 4, 0 # start with 16 byte alignment filled with zeros
        ret
        # nopl 0L(%[re]ax)
        # nopl 0L(%[re]ax,%[re]ax,1)
        # 0x0f, 0x1f, 0x80, 0x00, 0x00, 0x00, 0x00,
        # 0x0f, 0x1f, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00
        .align 4, 0x90
        ret

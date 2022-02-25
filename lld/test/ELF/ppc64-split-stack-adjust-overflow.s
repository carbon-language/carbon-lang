# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-unknown-linux %p/Inputs/ppc64-no-split-stack.s -o %t2.o

# RUN: not ld.lld %t1.o %t2.o -o /dev/null --defsym __morestack=0x10010000 2>&1 | \
# RUN: FileCheck -check-prefix=OVERFLOW %s
# RUN: not ld.lld %t1.o %t2.o -o /dev/null --defsym __morestack=0x10010000 \
# RUN: -split-stack-adjust-size 4097 2>&1 | FileCheck -check-prefix=OVERFLOW %s
# RUN: ld.lld %t1.o %t2.o -o %t --defsym __morestack=0x10010000 -split-stack-adjust-size 4096
# RUN: llvm-objdump -d %t | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=powerpc64-unknown-linux %p/Inputs/ppc64-no-split-stack.s -o %t2.o

# RUN: not ld.lld %t1.o %t2.o -o /dev/null --defsym __morestack=0x10010000 2>&1 | \
# RUN: FileCheck -check-prefix=OVERFLOW %s
# RUN: not ld.lld %t1.o %t2.o -o /dev/null --defsym __morestack=0x10010000 \
# RUN: -split-stack-adjust-size 4097 2>&1 | FileCheck -check-prefix=OVERFLOW %s
# RUN: ld.lld %t1.o %t2.o -o %t --defsym __morestack=0x10010000 -split-stack-adjust-size 4096
# RUN: llvm-objdump -d %t | FileCheck %s

# OVERFLOW: error: {{.*}}.o:(function caller: .text+0x8): split-stack prologue adjustment overflows

        .p2align    2
        .global caller
        .type caller, @function
caller:
.Lcaller_gep:
    addis 2, 12, .TOC.-.Lcaller_gep@ha
    addi 2, 2, .TOC.-.Lcaller_gep@l
    .localentry caller, .-caller
    ld 0, -0x7040(13)
    addis 12, 1, -32768
    addi  12, 12, 4096
    cmpld 7, 12, 0
    blt- 7, .Lcaller_alloc_more
.Lcaller_body:
    mflr 0
    std 0, 16(1)
    stdu 1, -32(1)
    bl nss_callee
    addi 1, 1, 32
    ld 0, 16(1)
    mtlr 0
    blr
.Lcaller_alloc_more:
    mflr 0
    std 0, 16(1)
    bl __morestack
    ld 0, 16(1)
    mtlr 0
    blr
    b .Lcaller_body
        .size caller, .-caller

# CHECK-LABEL: caller
# CHECK:      ld 0, -28736(13)
# CHECK-NEXT: addis 12, 1, -32768
# CHECK-NEXT: nop
# CHECK-NEXT: cmpld 7, 12, 0
# CHECK-NEXT: bt- 28, 0x10010204

.section        .note.GNU-split-stack,"",@progbits

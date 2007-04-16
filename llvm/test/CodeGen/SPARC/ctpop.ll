; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=sparc -mattr=v9 -enable-sparc-v9-insts
; RUN: llvm-upgrade < %s | llvm-as | llc -march=sparc -mattr=-v9 | \
; RUN:   not grep popc
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=sparc -mattr=v9 -enable-sparc-v9-insts | grep popc

declare uint %llvm.ctpop.i32(uint)
uint %test(uint %X) {
        %Y = call uint %llvm.ctpop.i32(uint %X)
        ret uint %Y
}


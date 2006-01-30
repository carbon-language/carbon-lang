; RUN: llvm-as < %s | llc -march=sparcv8 -mattr=-v9 &&
; RUN: llvm-as < %s | llc -march=sparcv8 -mattr=v9 -enable-sparc-v9-insts &&
; RUN: llvm-as < %s | llc -march=sparcv8 -mattr=-v9 | not grep popc &&
; RUN: llvm-as < %s | llc -march=sparcv8 -mattr=v9 -enable-sparc-v9-insts | grep popc

declare uint %llvm.ctpop.i32(uint)
uint %test(uint %X) {
        %Y = call uint %llvm.ctpop.i32(uint %X)
        ret uint %Y
}


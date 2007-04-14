; This test case is reduced from llvmAsmParser.cpp
; The optimizer should not remove the cast here.
; RUN: llvm-upgrade %s -o - | llvm-as | opt -instcombine | llvm-dis | \
; RUN:    grep sext.*i32

bool %test(short %X) {
    %A = cast short %X to uint
    %B = setgt uint %A, 1330
    ret bool %B
}

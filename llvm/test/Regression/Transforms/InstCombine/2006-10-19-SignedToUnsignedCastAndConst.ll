; This test case is reduced from llvmAsmParser.cpp
; The optimizer should not remove the cast here.
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | grep 'sext.*int'

bool %test(short %X) {
    %A = cast short %X to uint
    %B = setgt uint %A, 1330
    ret bool %B
}

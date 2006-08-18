; RUN: opt -analyze %s -datastructure-gc -dsgc-check-flags=x:IA

; ModuleID = 'bug3.bc'
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"


%struct.c99 = type {
        uint,
        uint,
        [0 x sbyte*] }

implementation   ; Functions:


void %foo(%struct.c99* %x) {
entry:
%B1 = getelementptr %struct.c99* %x, long 0, uint 2, uint 1
%B2 = getelementptr %struct.c99* %x, long 0, uint 2, uint 2
ret void
}

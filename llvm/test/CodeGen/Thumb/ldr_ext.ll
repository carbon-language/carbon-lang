; RUN: llvm-as < %s | llc -march=thumb | FileCheck %s

define i32 @test1(i8* %v.pntr.s0.u1) {
; CHECK: test1:
; CHECK: ldrb
    %tmp.u = load i8* %v.pntr.s0.u1
    %tmp1.s = zext i8 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test2(i16* %v.pntr.s0.u1) {
; CHECK: test2:
; CHECK: ldrh
    %tmp.u = load i16* %v.pntr.s0.u1
    %tmp1.s = zext i16 %tmp.u to i32
    ret i32 %tmp1.s
}

define i32 @test3(i8* %v.pntr.s1.u0) {
; CHECK: test3:
; CHECK: ldrb 
; CHECK: sxtb
   %tmp.s = load i8* %v.pntr.s1.u0
    %tmp1.s = sext i8 %tmp.s to i32
    ret i32 %tmp1.s
}

define i32 @test4() {
; CHECK: test4:
; CHECK: movs
; CHECK: ldrsh 
    %tmp.s = load i16* null
    %tmp1.s = sext i16 %tmp.s to i32
    ret i32 %tmp1.s
}

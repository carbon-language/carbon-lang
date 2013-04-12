; RUN: llc -march=mips -mattr=dsp < %s | FileCheck %s

; CHECK: test_lbux:
; CHECK: lbux ${{[0-9]+}}

define zeroext i8 @test_lbux(i8* nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i8* %b, i32 %i
  %0 = load i8* %add.ptr, align 1
  ret i8 %0
}

; CHECK: test_lhx:
; CHECK: lhx ${{[0-9]+}}

define signext i16 @test_lhx(i16* nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i16* %b, i32 %i
  %0 = load i16* %add.ptr, align 2
  ret i16 %0
}

; CHECK: test_lwx:
; CHECK: lwx ${{[0-9]+}}

define i32 @test_lwx(i32* nocapture %b, i32 %i) {
entry:
  %add.ptr = getelementptr inbounds i32* %b, i32 %i
  %0 = load i32* %add.ptr, align 4
  ret i32 %0
}

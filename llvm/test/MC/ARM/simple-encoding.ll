;RUN: llc -mtriple=armv7-apple-darwin -show-mc-encoding < %s | FileCheck %s


;FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;       should run on .s source files rather than using llc to generate the
;       assembly.

define i32 @foo(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: foo
; CHECK: trap                         @ encoding: [0xf0,0x00,0xf0,0x07]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]

  tail call void @llvm.trap()
  ret i32 undef
}

define i32 @f2(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: f2
; CHECK: add  r0, r1, r0              @ encoding: [0x00,0x00,0x81,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %add = add nsw i32 %b, %a
  ret i32 %add
}


define i32 @f3(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: f3
; CHECK: add  r0, r0, r1, lsl #3      @ encoding: [0x81,0x01,0x80,0xe0]
; CHECK: bx lr                        @ encoding: [0x1e,0xff,0x2f,0xe1]
  %mul = shl i32 %b, 3
  %add = add nsw i32 %mul, %a
  ret i32 %add
}

declare void @llvm.trap() nounwind

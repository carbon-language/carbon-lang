;RUN: llc -mtriple=armv7-apple-darwin -show-mc-encoding < %s | FileCheck %s


;FIXME: Once the ARM integrated assembler is up and going, these sorts of tests
;       should run on .s source files rather than using llc to generate the
;       assembly.

define i32 @foo(i32 %a, i32 %b) nounwind ssp {
entry:
; CHECK: foo
; CHECK: 0xf0,0x00,0xf0,0x07
; CHECK: 0x1e,0xff,0x2f,0xe1

  tail call void @llvm.trap()
  ret i32 undef
}

define i32 @f2(i32 %a, i32 %b) nounwind readnone ssp {
entry:
; CHECK: f2
; CHECK: 0x00,0x00,0x81,0xe0
; CHECK: 0x1e,0xff,0x2f,0xe1
  %add = add nsw i32 %b, %a
  ret i32 %add
}
declare void @llvm.trap() nounwind

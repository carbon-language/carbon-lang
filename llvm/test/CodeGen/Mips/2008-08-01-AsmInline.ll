; RUN: llc -march=mips < %s | FileCheck %s

%struct.DWstruct = type { i32, i32 }

define i32 @A0(i32 %u, i32 %v) nounwind  {
entry:
; CHECK: multu 
; CHECK: mflo
; CHECK: mfhi
  %asmtmp = tail call %struct.DWstruct asm "multu $2,$3", "={lo},={hi},d,d"( i32 %u, i32 %v ) nounwind
  %asmresult = extractvalue %struct.DWstruct %asmtmp, 0
  %asmresult1 = extractvalue %struct.DWstruct %asmtmp, 1    ; <i32> [#uses=1]
  %res = add i32 %asmresult, %asmresult1
  ret i32 %res
}

; RUN: llc -mtriple powerpc64-linux < %s | FileCheck %s

define void @foo()  {
  ret void
}
declare i32 @bar(i8*)

; CHECK-LABEL: {{^}}zed:
; CHECK:        addis 3, 2, .LC1@toc@ha
; CHECK-NEXT:   ld 3, .LC1@toc@l(3)
; CHECK-NEXT:   bl bar


; CHECK-LABEL: .section        .toc,"aw",@progbits
; CHECK:       .LC1:
; CHECK-NEXT:  .tc foo[TC],foo

define  void @zed() {
  call i32 @bar(i8* bitcast (void ()* @foo to i8*))
  ret void
}

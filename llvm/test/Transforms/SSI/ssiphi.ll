; RUN: opt < %s -ssi-everything -S | FileCheck %s

declare void @use(i32)
declare i32 @create()

define i32 @foo() {
entry:
  %x = call i32 @create()
  %y = icmp slt i32 %x, 10
  br i1 %y, label %T, label %F
T:
; CHECK: SSI_sigma 
  call void @use(i32 %x)
  br label %join
F:
; CHECK: SSI_sigma
  call void @use(i32 %x)
  br label %join
join:
; CHECK: SSI_phi
  ret i32 %x
}

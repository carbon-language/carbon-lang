; RUN: llc -march=mipsel < %s | FileCheck %s

; CHECK: seteq0:
; CHECK: sltiu ${{[0-9]+}}, $4, 1

define i32 @seteq0(i32 %a) {
entry:
  %cmp = icmp eq i32 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

; CHECK: setne0:
; CHECK: sltu ${{[0-9]+}}, $zero, $4

define i32 @setne0(i32 %a) {
entry:
  %cmp = icmp ne i32 %a, 0
  %conv = zext i1 %cmp to i32
  ret i32 %conv
}

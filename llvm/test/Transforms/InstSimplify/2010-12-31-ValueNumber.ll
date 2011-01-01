; RUN: opt < %s -instsimplify -S | FileCheck %s
define i1 @g(i32 %a) nounwind readnone {
; CHECK: @g
  %add = shl i32 %a, 1
  %mul = shl i32 %a, 1
  %cmp = icmp ugt i32 %add, %mul
  ret i1 %cmp
; CHECK: ret i1 false
}

; RUN: llc < %s -march=mipsel | FileCheck %s

@g1 = external global i32

define i32 @foo0(i32 %s) nounwind readonly {
entry:
; CHECK:     movn ${{[0-9]+}}, $zero
  %tobool = icmp ne i32 %s, 0
  %0 = load i32* @g1, align 4
  %cond = select i1 %tobool, i32 0, i32 %0
  ret i32 %cond
}

define i32 @foo1(i32 %s) nounwind readonly {
entry:
; CHECK:     movz ${{[0-9]+}}, $zero
  %tobool = icmp ne i32 %s, 0
  %0 = load i32* @g1, align 4
  %cond = select i1 %tobool, i32 %0, i32 0
  ret i32 %cond
}

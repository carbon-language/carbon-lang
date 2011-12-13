; RUN: true
; DISABLED: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s

@num = external global i32
@acc = external global i32
@val = external global i32

; CHECK: CONST32(#acc)
; CHECK: CONST32(#val)
; CHECK: CONST32(#num)

define void @foo() nounwind {
entry:
  %0 = load i32* @num, align 4
  %1 = load i32* @acc, align 4
  %mul = mul nsw i32 %0, %1
  %2 = load i32* @val, align 4
  %add = add nsw i32 %mul, %2
  store i32 %add, i32* @num, align 4
  ret void
}

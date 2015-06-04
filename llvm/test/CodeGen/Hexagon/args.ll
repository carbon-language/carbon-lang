; RUN: llc -march=hexagon -mcpu=hexagonv4 -disable-dfa-sched -disable-hexagon-misched < %s | FileCheck %s
; CHECK: memw(r29{{ *}}+{{ *}}#0){{ *}}={{ *}}#7
; CHECK: r1:0 = combine(#2, #1)
; CHECK: r3:2 = combine(#4, #3)
; CHECK: r5:4 = combine(#6, #5)


define void @foo() nounwind {
entry:
  call void @bar(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7)
  ret void
}

declare void @bar(i32, i32, i32, i32, i32, i32, i32)

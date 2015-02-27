; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s
; CHECK: word
; CHECK-NOT: combine(#0
; CHECK: jump bar

define void @word(i32* nocapture %a) nounwind {
entry:
  %0 = load i32* %a, align 4
  %1 = zext i32 %0 to i64
  %add.ptr = getelementptr inbounds i32, i32* %a, i32 1
  %2 = load i32* %add.ptr, align 4
  %3 = zext i32 %2 to i64
  %4 = shl nuw i64 %3, 32
  %ins = or i64 %4, %1
  tail call void @bar(i64 %ins) nounwind
  ret void
}

declare void @bar(i64)

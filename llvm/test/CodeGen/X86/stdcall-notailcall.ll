; RUN: llc -mtriple=i386-apple-darwin11 -O2 < %s | FileCheck %s

%struct.I = type { i32 (...)** }
define x86_stdcallcc void @bar(%struct.I* nocapture %this) ssp align 2 {
; CHECK-LABEL: bar:
; CHECK-NOT: jmp
; CHECK: retl $4
entry:
  tail call void @foo()
  ret void
}

define x86_thiscallcc void @test2(%struct.I*  %this, i32 %a) {
; CHECK-LABEL: test2:
; CHECK: calll _foo
; CHECK: retl $4
  tail call void @foo()
  ret void
}

declare void @foo()

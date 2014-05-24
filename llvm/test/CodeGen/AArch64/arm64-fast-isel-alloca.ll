; This test should cause the TargetMaterializeAlloca to be invoked
; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

%struct.S1Ty = type { i64 }
%struct.S2Ty = type { %struct.S1Ty, %struct.S1Ty }

define void @takeS1(%struct.S1Ty* %V) nounwind {
entry:
  %V.addr = alloca %struct.S1Ty*, align 8
  store %struct.S1Ty* %V, %struct.S1Ty** %V.addr, align 8
  ret void
}

define void @main() nounwind {
entry:
; CHECK: main
; CHECK: mov x29, sp
; CHECK: mov x[[REG:[0-9]+]], sp
; CHECK-NEXT: orr x[[REG1:[0-9]+]], xzr, #0x8
; CHECK-NEXT: add x0, x[[REG]], x[[REG1]]
  %E = alloca %struct.S2Ty, align 4
  %B = getelementptr inbounds %struct.S2Ty* %E, i32 0, i32 1
  call void @takeS1(%struct.S1Ty* %B)
  ret void
}

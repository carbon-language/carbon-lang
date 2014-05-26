; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

; This file contains tests for the AArch64 load/store optimizer.

%struct.A = type { %struct.B, %struct.C }
%struct.B = type { i8*, i8*, i8*, i8* }
%struct.C = type { i32, i32 }

; Check the following transform:
;
; ldr w1, [x0, #32]
;  ...
; add x0, x0, #32
;  ->
; ldr w1, [x0, #32]!

define void @foo(%struct.A* %ptr) nounwind {
; CHECK-LABEL: foo
; CHECK: ldr w{{[0-9]+}}, [x{{[0-9]+}}, #32]!
entry:
  %a = getelementptr inbounds %struct.A* %ptr, i64 0, i32 1, i32 0
  %add = load i32* %a, align 4
  br label %bar
bar:
  %c = getelementptr inbounds %struct.A* %ptr, i64 0, i32 1
  tail call void @bar(%struct.C* %c, i32 %add)
  ret void
}

declare void @bar(%struct.C*, i32)


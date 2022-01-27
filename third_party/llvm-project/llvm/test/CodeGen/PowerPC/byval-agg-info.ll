; RUN: llc -verify-machineinstrs < %s -print-after=prologepilog >%t 2>&1 && FileCheck <%t %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.anon = type { i32, i32 }

declare void @foo(%struct.anon* %v)
define void @test(i32 %a, i32 %b, %struct.anon* byval(%struct.anon) nocapture %v) {
entry:
  call void @foo(%struct.anon* %v)
  ret void
}

; Make sure that the MMO on the store has no offset from the byval
; variable itself (we used to have (store (s64) into %ir.v + 64)).
; CHECK: STD killed renamable $x5, 176, $x1 :: (store (s64) into %ir.v, align 16)


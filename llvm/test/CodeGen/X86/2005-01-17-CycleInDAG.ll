; This testcase was distilled from 132.ijpeg.  Bsaically we cannot fold the
; load into the sub instruction here as it induces a cycle in the dag, which
; is invalid code (there is no correct way to order the instruction).  Check
; that we do not fold the load into the sub.

; RUN: llc < %s -mtriple=i686-- | FileCheck %s

@GLOBAL = external dso_local global i32

define i32 @test(i32* %P1, i32* %P2, i32* %P3) nounwind {
; CHECK-LABEL: test:
entry:
  %L = load i32, i32* @GLOBAL
  store i32 12, i32* %P2
  %Y = load i32, i32* %P3
  %Z = sub i32 %Y, %L
  ret i32 %Z
; CHECK-NOT: {{sub.*GLOBAL}}
}


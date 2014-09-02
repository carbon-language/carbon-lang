; RUN: opt < %s -cfl-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; Derived from BasicAA/2010-09-15-GEP-SignedArithmetic.ll

target datalayout = "e-p:32:32:32"

; CHECK: 1 partial alias response

define i32 @test(i32* %tab, i32 %indvar) nounwind {
  %tmp31 = mul i32 %indvar, -2
  %tmp32 = add i32 %tmp31, 30
  %t.5 = getelementptr i32* %tab, i32 %tmp32
  %loada = load i32* %tab
  store i32 0, i32* %t.5
  %loadb = load i32* %tab
  %rval = add i32 %loada, %loadb
  ret i32 %rval
}

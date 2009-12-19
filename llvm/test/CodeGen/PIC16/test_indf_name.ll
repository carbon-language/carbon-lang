; RUN: llvm-as < %s | llc -march=pic16 | FileCheck %s

@pi = common global i16* null, align 1            ; <i16**> [#uses=1]

define void @foo() nounwind {
entry:
  %tmp = load i16** @pi                           ; <i16*> [#uses=1]
  store i16 1, i16* %tmp
; CHECK: movwi {{[0-1]}}[INDF{{[0-1]}}]
; CHECK: movwi {{[0-1]}}[INDF{{[0-1]}}]
  ret void
}

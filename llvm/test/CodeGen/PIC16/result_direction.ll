; RUN: llvm-as < %s | llc -march=pic16 | FileCheck %s

@a = common global i16 0, align 1                 ; <i16*> [#uses=2]

define void @foo() nounwind {
entry:
  %tmp = load i16* @a                             ; <i16> [#uses=1]
  %add = add nsw i16 %tmp, 1                      ; <i16> [#uses=1]
  store i16 %add, i16* @a
;CHECK: movlw 1
;CHECK: addwf @a + 0, F
  ret void
}

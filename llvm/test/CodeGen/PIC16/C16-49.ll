;RUN: llvm-as < %s | llc -march=pic16

@aa = global i16 55, align 1                      ; <i16*> [#uses=1]
@bb = global i16 44, align 1                      ; <i16*> [#uses=1]
@PORTD = external global i8                       ; <i8*> [#uses=1]

define void @foo() nounwind {
entry:
  %tmp = volatile load i16* @aa                   ; <i16> [#uses=1]
  %tmp1 = volatile load i16* @bb                  ; <i16> [#uses=1]
  %sub = sub i16 %tmp, %tmp1                      ; <i16> [#uses=1]
  %conv = trunc i16 %sub to i8                    ; <i8> [#uses=1]
  store i8 %conv, i8* @PORTD
  ret void
}

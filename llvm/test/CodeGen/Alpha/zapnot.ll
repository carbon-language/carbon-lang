; Make sure this testcase codegens to the bic instruction
; RUN: llc < %s -march=alpha | grep zapnot


define zeroext i16 @foo(i64 %y)  {
entry:
        %tmp.1 = trunc i64 %y to i16         ; <ushort> [#uses=1]
        ret i16 %tmp.1
}
